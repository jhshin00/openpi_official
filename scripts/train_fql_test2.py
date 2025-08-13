import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state_fql(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState_AC, Any]:
    """Initialize TrainState_AC for Pi0FQL, loading base weights if provided."""
    # create optimizer
    tx_c = _optimizer.create_optimizer(config.optimizer, config.lr_schedule_critic, weight_decay_mask=None)
    tx_a = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    # tx_c = optax.chain(optax.clip_by_global_norm(10.0),_optimizer.create_optimizer(config.optimizer, config.lr_schedule_critic, weight_decay_mask=None))
    # tx_a = optax.chain(optax.clip_by_global_norm(10.0),_optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None))
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState_AC:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)  # Pi0FQL

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # convert frozen params to bfloat16
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        critic_params = params.filter(lambda path, _: path[0] == "critic")
        actor_params = params.filter(lambda path, _: path[0] == "actor")
        critic_target_params = params.filter(lambda path, _: path[0] == "critic_target")

        return training_utils.TrainState_AC(
            step=0,
            critic_params=critic_params,
            critic_target_params=critic_target_params,
            actor_params=actor_params,
            model_def=nnx.graphdef(model),
            tx_critic=tx_c,
            tx_actor=tx_a,
            critic_opt_state=tx_c.init(critic_params.filter(config.trainable_filter)),
            actor_opt_state=tx_a.init(actor_params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    # infer shape & sharding
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding


    # full_dict = train_state_shape.params.to_pure_dict()
    # loaded_params = config.weight_loader.load(full_dict)
    # #at.check_pytree_equality(expected=full_dict, got=loaded_params, check_shapes=True, check_dtypes=True)
    # flat_params = traverse_util.flatten_dict(loaded_params)
    # flat actor = {
    #     k: v for k, v in flat_params.items() if not is instance(v, jax.ShapeDtypeStruct)
    # }
    # flat_actor = {
    #     path: val for path, val in flat.items() if path[0] == "actor" and not isinstance(val, jax.ShapeDtypeStruct)
    # }

    # partial_params = traverse_util.unflatten_dict(flat_actor)
    # replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    
    actor_shape = train_state_shape.actor_params.to_pure_dict()
    loaded = _load_weights_and_validate(config.weight_loader, actor_shape["actor"])
    partial_params = {'actor': loaded}


    # ########################################
    # logging.info("Checkpoint loading check")
    # rand_state = init(init_rng, None)
    # load_state = init(init_rng, partial_params)
    # diffs = jax.tree_util.tree_map(
    #     lambda a, b: jnp.max(jnp.abs(a - b)),
    #     rand_state.actor_params, load_state.actor_params
    # )
    # flat = traverse_util.flatten_dict(diffs.to_pure_dict(), sep="/")
    # for path, diff in flat.items():
    #     try:
    #         val = float(jax.device_get(diff).item())
    #         logging.info(f"[Param diff] actor/{path}: max|Δ| = {val:.6f}")
    #     except:
    #         logging.info(f"[Param diff] actor/{path}: {diff}")
    # #########################################
    
    
    # partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.actor_params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    # ##################################################################################
    # logging.info("Checkpoint loading check")
    # rand_state = init(init_rng, None)
    # diffs = jax.tree_util.tree_map(
    #     lambda a, b: jnp.max(jnp.abs(a - b)),
    #     rand_state.actor_params, train_state.actor_params
    # )
    # del rand_state
    # flat = traverse_util.flatten_dict(diffs.to_pure_dict(), sep="/")
    # for path, diff in flat.items():
    #     try:
    #         val = float(jax.device_get(diff).item())
    #         logging.info(f"[Param diff] {path}: max|Δ| = {val:.6f}")
    #     except:
    #         logging.info(f"[Param diff] {path}: {diff}")
    # del diffs
    # del flat
    # ##################################################################################
    
    # logging.info("Checkpoint loading check")
    # rand_state  = init(init_rng, None)                 # 전부 랜덤
    #       # 일부만 로드

    # # diff = |rand - train|
    # diffs = jax.tree_util.tree_map(
    #     lambda a, b: jnp.max(jnp.abs(a - b)), 
    #     rand_state.actor_params, 
    #     train_state.actor_params
    # )

    # flat_diffs = traverse_util.flatten_dict(diffs.to_pure_dict(), sep="/")

    # missing, loaded = [], []
    # for path, diff in flat_diffs.items():
    #     # bfloat16 정밀 오차 감안, 1e-7 미만이면 0으로 취급
    #     diff_val = float(jax.device_get(diff))
    #     if diff_val < 1e-7:
    #         missing.append(path)        # 로드 실패
    #     else:
    #         loaded.append(path)         # 로드 OK

    # logging.info(f"✓  loaded params : {len(loaded):5d}")
    # logging.info(f"✗ missing params: {len(missing):5d}")

    # for p in missing:
    #     logging.info(f"[MISSING] {p}")
    
    return train_state, state_sharding


@at.typecheck
def train_step_fql(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState_AC,
    batch: tuple[_model.Observation, _model.Actions, at.Float[at.Array, "b H"], at.Bool[at.Array, "b"],  _model.Observation, at.Bool[at.Array, "b H"], at.Bool[at.Array, "b H"]],
) -> tuple[training_utils.TrainState_AC, dict[str, at.Array]]:
    
    """Single training step: compute loss from Pi0FQL and update parameters."""
    model = nnx.merge(state.model_def, state.actor_params, state.critic_params, state.critic_target_params)
    model.train()
    
    @at.typecheck
    def critic_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        obs: _model.Observation,
        acts: _model.Actions,
        rews: at.Float[at.Array, "b H"],
        term: at.Bool[at.Array, "b"],
        next_obs: _model.Observation,
        reward_is_pad: at.Bool[at.Array, "b H"],
        # actions_is_pad: at.Bool[at.Array, "b H"],
    ):
        return model.critic_chunk_loss(rng, obs, acts, rews, term, next_obs, reward_is_pad, train=True)
    
    @at.typecheck
    def actor_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        obs: _model.Observation,
        acts: _model.Actions,
        rews: at.Float[at.Array, "b H"],
        term: at.Bool[at.Array, "b"],
        next_obs: _model.Observation,
        # actions_is_pad: at.Bool[at.Array, "b H"],
    ):
        # Use actor_chunk_loss and pass actions_is_pad for consistency
        _, (flow_loss, q_loss) = model.actor_chunk_loss(rng, obs, acts, rews, term, next_obs, train=True)
        
        # Adaptive Q-loss coefficient with warmup to prevent BPTT explosion
        t = state.step.astype(jnp.float32)
        
        rl_coef = jnp.where(
            t < config.actor_warmup_steps,
            0.0,
            jnp.minimum(1.0, (t-config.actor_warmup_steps) / config.rl_warmup_steps) * config.alpha_q
        )
        
        flow_coef = 1.0
        actor_loss = flow_coef * flow_loss + rl_coef * q_loss
        return actor_loss, (flow_coef, rl_coef, flow_loss, q_loss)
    

    train_rng = jax.random.fold_in(rng, state.step)
    ###########################################################################################################
    train_rng, train_rng_actor, train_rng_critic = jax.random.split(train_rng, 3)
    ###########################################################################################################

    obs, acts, rews, term, next_obs, _, reward_is_pad = batch

    diff_state_critic = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=="critic", config.trainable_filter))
    grad_critic_fn = nnx.value_and_grad(critic_loss_fn, argnums=diff_state_critic)

    diff_state_actor = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=="actor", config.trainable_filter))
    grad_actor_fn = nnx.value_and_grad(actor_loss_fn, argnums=diff_state_actor, has_aux=True)

    critic_loss, critic_grads = grad_critic_fn(model, train_rng_critic, obs, acts, rews, term, next_obs, reward_is_pad)
    critic_params = state.critic_params.filter(config.trainable_filter)
    updates_c, new_critic_opt = state.tx_critic.update(critic_grads, state.critic_opt_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates_c)
    # nnx.update(model, new_critic_params)
    
    critic_target_params = state.critic_target_params.filter(config.trainable_filter)
    new_target_critic_param_subtree = jax.tree.map(
        lambda p, tp: p * config.tau_target + tp * (1 - config.tau_target),
        new_critic_params['critic'],
        critic_target_params['critic_target'],
    )

    new_critic_target_params = nnx.statelib.State(
        {'critic_target': new_target_critic_param_subtree.to_pure_dict()}
    )
    # nnx.update(model, new_critic_target_params)

    (actor_loss, (flow_coef, rl_coef, flow_loss, q_loss)), actor_grads = grad_actor_fn(model, train_rng_actor, obs, acts, rews, term, next_obs)
    actor_params = state.actor_params.filter(config.trainable_filter)
    updates_a, new_actor_opt = state.tx_actor.update(actor_grads, state.actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates_a)

    nnx.update(model, new_actor_params, new_critic_params, new_critic_target_params)

    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        critic_params=new_params.filter(lambda path, _: path[0] == "critic"),
        critic_target_params=new_params.filter(lambda path, _: path[0] == "critic_target"),
        actor_params=new_params.filter(lambda path, _: path[0] == "actor"),
        critic_opt_state=new_critic_opt,
        actor_opt_state=new_actor_opt,
    )

    ###########################################################################################################
    
    
    # if state.ema_decay is not None:
    #     new_state = dataclasses.replace(
    #         new_state,
    #         ema_params=jax.tree_map(
    #             lambda old,
    #             new: state.ema_decay * old + (1 - state.ema_decay) * new,
    #             state.ema_params,
    #             new_params,
    #         ),
    #     )

    critic_grad_norm = optax.global_norm(critic_grads)
    actor_grad_norm = optax.global_norm(actor_grads)
    grad_norm = jnp.sqrt(critic_grad_norm**2 + actor_grad_norm**2)
    loss = critic_loss + actor_loss
    
    info = {
        "flow_loss": flow_loss,
        "actor_loss": actor_loss,
        "q_loss": q_loss,
        "critic_loss": critic_loss,
        "total_loss": loss,
        "grads": grad_norm,
        "actor_grads": actor_grad_norm,
        "critic_grads": critic_grad_norm,
        "flow_coef" : flow_coef,
        "rl_coef": rl_coef,
    }

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state_fql(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state - actor:\n{training_utils.array_tree_to_info(train_state.actor_params)}")
    logging.info(f"Initialized train state - critic:\n{training_utils.array_tree_to_info(train_state.critic_params)}")
    logging.info(f"Initialized train state - critic_target:\n{training_utils.array_tree_to_info(train_state.critic_target_params)}")

    if resuming:
        train_state = _checkpoints.restore_state_fql(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step_fql, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            msg = ", ".join(f"{k}={v:.4f}" for k, v in reduced.items())
            pbar.write(f"Step {step}: {msg}")
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)
        # _, actions, rewards, terminal, _ = batch
        # # logging.info(f"actions : {actions[0,0,:8]}")
        # # logging.info(f"actions : {actions[0,0,8:16]}")
        # # logging.info(f"actions : {actions[0,0,16:24]}")
        # # logging.info(f"actions : {actions[0,0,24:32]}")

        # assert jnp.allclose(rewards, terminal.astype(jnp.float32), atol=1e-6), "rewards != terminal"

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state_fql(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    

if __name__ == "__main__":
    main(_config.cli())
