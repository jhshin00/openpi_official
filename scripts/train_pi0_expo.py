import dataclasses
import functools
import logging
import platform
from typing import Any

import tracemalloc

import os
import sys
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

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
import openpi.training.expo_config as _config
import openpi.models.tokenizer as _tokenizer
# import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from openpi.training.expo_buffer import TrajReplayBuffer
from openpi.training.retriever import SimpleRetriever
from openpi.training.expo_train_utils import (
    get_libero_env,
    collect_trajectory,
    add_online_data_to_buffer,
    perform_control_eval,
    create_expo_obs_space,
    create_expo_action_space,
)

from libero.libero import benchmark

# Suppress JAX and Flax deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")



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
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainStatePi0Expo, Any]:
    tx_actor = _optimizer.create_optimizer(config.optimizer_actor, config.lr_schedule_actor, weight_decay_mask=None)
    tx_critic = _optimizer.create_optimizer(config.optimizer_critic, config.lr_schedule_critic, weight_decay_mask=None)
    tx_edit_actor = _optimizer.create_optimizer(config.optimizer_edit_actor, config.lr_schedule_edit_actor, weight_decay_mask=None)
    tx_temp = _optimizer.create_optimizer(config.optimizer_temp, config.lr_schedule_temp, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        
        critic_params = params.filter(lambda path, _: path[0] == "critic")
        target_critic_params = params.filter(lambda path, _: path[0] == "target_critic")
        actor_params = params.filter(lambda path, _: path[0] == "actor")
        edit_actor_params = params.filter(lambda path, _: path[0] == "edit_actor")
        temp_params = params.filter(lambda path, _: path[0] == "temp")
        
        
        return training_utils.TrainStatePi0Expo(
            step=0,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            actor_params=actor_params,
            edit_actor_params=edit_actor_params,
            temp_params=temp_params,
            model_def=nnx.graphdef(model),
            critic_opt_state=tx_critic.init(critic_params.filter(config.trainable_filter)),
            actor_opt_state=tx_actor.init(actor_params.filter(config.trainable_filter)),
            edit_actor_opt_state=tx_edit_actor.init(edit_actor_params.filter(config.trainable_filter)),
            temp_opt_state=tx_temp.init(temp_params.filter(config.trainable_filter)),
            tx_critic=tx_critic,
            tx_actor=tx_actor,
            tx_edit_actor=tx_edit_actor,
            tx_temp=tx_temp,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding
    
    actor_shape = train_state_shape.actor_params.to_pure_dict()
    loaded = _load_weights_and_validate(config.weight_loader, actor_shape["actor"])
    partial_params = {'actor': loaded}
    

    # partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainStatePi0Expo,
    batch: tuple[_model.Observation, _model.Actions, at.Float[at.Array, "b H"], _model.Observation, at.Bool[at.Array, "b"]],
    retrieval_now: at.Array | None,
    retrieval_next: at.Array | None,
) -> tuple[training_utils.TrainStatePi0Expo, dict[str, at.Array]]:
    model = nnx.merge(
        state.model_def,
        state.critic_params,
        state.target_critic_params,
        state.actor_params,
        state.edit_actor_params,
        state.temp_params,
    )
    model.train()
    
    @at.typecheck
    def actor_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ):
        chunked_loss = model.actor_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)
    
    @at.typecheck
    def critic_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        rewards: at.Float[at.Array, "b H"],
        next_observation: _model.Observation,
        masks: at.Bool[at.Array, "b"],
        retrieval_actions: at.Float[at.Array, "*b k ah ad"],
    ):
        return model.critic_loss(rng, observation, actions, rewards, next_observation, masks, train=True, retrieval_actions=retrieval_actions)
    

    @at.typecheck
    def edit_actor_loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        retrieval_actions: at.Float[at.Array, "*b k ah ad"],
    ):
        loss, entropy = model.edit_actor_loss(rng, observation, actions, train=True, retrieval_actions=retrieval_actions)
        return loss, entropy
    
    @at.typecheck
    def temperature_loss_fn(
        model: _model.BaseModel,
        entropy: at.Float[at.Array, ""],
    ):
        return model.temperature_loss(entropy)

    train_rng = jax.random.fold_in(rng, state.step)
    train_rng, train_rng_actor, train_rng_critic, train_rng_edit_actor = jax.random.split(train_rng, 4)
    observation, actions, rewards, next_observation, masks = batch

    diff_state_critic = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=='critic', config.trainable_filter))
    grad_critic_fn = nnx.value_and_grad(critic_loss_fn, argnums=diff_state_critic)
    diff_state_actor = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=='actor', config.trainable_filter))
    grad_actor_fn = nnx.value_and_grad(actor_loss_fn, argnums=diff_state_actor)
    diff_state_edit_actor = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=='edit_actor', config.trainable_filter))
    grad_edit_actor_fn = nnx.value_and_grad(edit_actor_loss_fn, argnums=diff_state_edit_actor, has_aux=True)
    diff_state_temp = nnx.DiffState(0, nnx.All(lambda path, _: path[0]=='temp', config.trainable_filter))
    grad_temp_fn = nnx.value_and_grad(temperature_loss_fn, argnums=diff_state_temp)

    critic_loss, critic_grads = grad_critic_fn(model, train_rng_critic, observation, actions, rewards, next_observation, masks, retrieval_next)
    critic_params = state.critic_params.filter(config.trainable_filter)
    updates_c, new_critic_opt_state = state.tx_critic.update(critic_grads, state.critic_opt_state, critic_params)
    new_critic_params = optax.apply_updates(critic_params, updates_c)

    target_critic_params = state.target_critic_params.filter(config.trainable_filter)
    new_target_critic_param_subtree = jax.tree.map(
        lambda p, tp: p * config.tau_target + tp * (1 - config.tau_target),
        new_critic_params['critic'],
        target_critic_params['target_critic'],
    )

    new_target_critic_params = nnx.statelib.State(
        {'target_critic': new_target_critic_param_subtree.to_pure_dict()}
    )

    actor_loss, actor_grads = grad_actor_fn(model, train_rng_actor, observation, actions)
    actor_params = state.actor_params.filter(config.trainable_filter)
    updates_a, new_actor_opt_state = state.tx_actor.update(actor_grads, state.actor_opt_state, actor_params)
    new_actor_params = optax.apply_updates(actor_params, updates_a)


    (edit_actor_loss, entropy), edit_actor_grads = grad_edit_actor_fn(model, train_rng_edit_actor, observation, actions, retrieval_now)
    edit_actor_params = state.edit_actor_params.filter(config.trainable_filter)
    updates_e, new_edit_actor_opt_state = state.tx_edit_actor.update(edit_actor_grads, state.edit_actor_opt_state, edit_actor_params)
    new_edit_actor_params = optax.apply_updates(edit_actor_params, updates_e)

    temp_loss, temp_grads = grad_temp_fn(model, entropy)
    temp_params = state.temp_params.filter(config.trainable_filter)
    updates_t, new_temp_opt_state = state.tx_temp.update(temp_grads, state.temp_opt_state, temp_params)
    new_temp_params = optax.apply_updates(temp_params, updates_t)

    nnx.update(model, new_actor_params, new_critic_params, new_target_critic_params, new_edit_actor_params, new_temp_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        critic_params=new_params.filter(lambda path, _: path[0] == "critic"),
        target_critic_params=new_params.filter(lambda path, _: path[0] == "target_critic"),
        actor_params=new_params.filter(lambda path, _: path[0] == "actor"),
        edit_actor_params=new_params.filter(lambda path, _: path[0] == "edit_actor"),
        temp_params=new_params.filter(lambda path, _: path[0] == "temp"),
        critic_opt_state=new_critic_opt_state,
        actor_opt_state=new_actor_opt_state,
        edit_actor_opt_state=new_edit_actor_opt_state,
        temp_opt_state=new_temp_opt_state,
    )

    info = {
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "edit_actor_loss": edit_actor_loss,
        "temp_loss": temp_loss,
        "actor_grad_norm": optax.global_norm(actor_grads),
        "critic_grad_norm": optax.global_norm(critic_grads),
        "edit_actor_grad_norm": optax.global_norm(edit_actor_grads),
        "temp_grad_norm": optax.global_norm(temp_grads),
        "entropy": entropy,
    }

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    
    # Start memory tracing for debugging
    tracemalloc.start()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng, traj_rng, eval_rng = jax.random.split(rng, 4)

    mesh = sharding.make_mesh(config.fsdp_devices)

    # data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = replicated_sharding

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    replay_buffer = TrajReplayBuffer(
        observation_space=create_expo_obs_space(),
        action_space=create_expo_action_space(),
        capacity=config.buffer.capacity_total,
        use_offline_data=config.use_offline_data,
        libero_data_dir=config.libero_data_dir,
        offline_dataset_subset_num=config.offline_dataset_subset_num,
        batch_offline_ratio=config.buffer.batch_offline_ratio,
        success_memory_per_task=config.buffer.success_memory_per_task,
        eviction=config.buffer.eviction,
    )
    data_iter = replay_buffer.get_iterator(
        config.batch_size,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[config.libero_task_suite]()
    num_tasks = task_suite.get_num_tasks()
    logging.info(f"Task suite: {config.libero_task_suite}, Number of tasks: {num_tasks}")
    
    # 모든 task들의 description을 출력
    for i in range(num_tasks):
        task = task_suite.get_task(i)
        logging.info(f"Task {i}: {task.language}")

    
    env = None
    task_description = None
    eval_env = None

    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    # images_to_log = _preview_images_from_obs(batch[0], k=5)
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state - actor:\n{training_utils.array_tree_to_info(train_state.actor_params)}")
    logging.info(f"Initialized train state - critic:\n{training_utils.array_tree_to_info(train_state.critic_params)}")
    logging.info(f"Initialized train state - critic_target:\n{training_utils.array_tree_to_info(train_state.target_critic_params)}")
    logging.info(f"Initialized train state - edit_actor:\n{training_utils.array_tree_to_info(train_state.edit_actor_params)}")
    logging.info(f"Initialized train state - temp:\n{training_utils.array_tree_to_info(train_state.temp_params)}")

    # === [ADD] 평가용 모델 빌더 (critic trunk 임베딩에 사용) ===
    def _build_eval_model(state):
        model = nnx.merge(
            state.model_def,
            state.critic_params,
            state.target_critic_params,
            state.actor_params,
            state.edit_actor_params,
            state.temp_params,
        )
        model.eval()
        return model

    # === [ADD] Retriever 초기화 ===
    retriever = None
    if config.retrieval.enabled:
        model_eval = _build_eval_model(train_state)
        retriever = SimpleRetriever(
            H=config.action_horizon,
            A=config.action_dim,
            success_only=config.retrieval.success_only,
            topk=config.retrieval.topk,
            alpha_q=config.retrieval.alpha_q,
        )
        # 오프라인으로 초기 인덱스 구축
        retriever.build_from_buffer(model_eval, replay_buffer)

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding,data_sharding,data_sharding),
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

    tokenizer = _tokenizer.PaligemmaTokenizer(config.max_token_len)

    infos = []
    for step in pbar:
        # --- tracemalloc 디버깅 코드 시작 ---
        if 'snapshot_before' not in locals():
            snapshot_before = tracemalloc.take_snapshot()

        if step % 500 == 0 and step > 0: # 500 스텝마다 메모리 증가량 비교
            print(f"\n--- Memory Profile at Step {step} ---")
            snapshot_after = tracemalloc.take_snapshot()
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

            print("[ Top 10 memory increases ]")
            for stat in top_stats[:10]:
                print(stat)

            snapshot_before = snapshot_after # 다음 비교를 위해 현재 스냅샷을 저장
        # --- tracemalloc 디버깅 코드 끝 ---


        if step >= config.offline_steps:
            if step % config.rollout_interval == 0:
                if step % (config.env_reuse_frequency*config.rollout_interval) == 0 or env is None:
                    traj_rng, task_rng = jax.random.split(traj_rng)
                    task_id = jax.random.randint(task_rng, (), 0, num_tasks)
                    task = task_suite.get_task(task_id)
                    env, task_description = get_libero_env(task, 256, config.seed)
                    logging.info(f"Online data collection at step {step}: task_id={task_id}, task_description='{task_description}'")
                    # eval_env = env

                traj_rng = jax.random.fold_in(traj_rng, step)
                traj = collect_trajectory(traj_rng, config, train_state, env, task_description, tokenizer)
                add_online_data_to_buffer(traj, replay_buffer)
        
        else:
            traj = {
                'is_success': False,
                'episode_return': 0.0,
                'episode_length': 0,
                'env_steps': 0
            }
        
        # batch = shard_batch_strict(batch, data_sharding, mesh)
        # _check_batch_shapes(batch, config.batch_size, config.action_horizon, config.action_dim)
        # debug_print_shardings(batch)

        if retriever is not None and config.retrieval.enabled:
            do_refresh = False
            # 1) 주기적 리프레시
            if step % config.retrieval.refresh_every == 0:
                do_refresh = True
            # 2) 방금 온라인 데이터 들어왔으면 즉시 리프레시(너무 자주면 빼도 됨)
            if step >= config.offline_steps and (step % config.rollout_interval == 0):
                do_refresh = True

            if do_refresh:
                model_eval = _build_eval_model(train_state)
                retriever.build_from_buffer(model_eval, replay_buffer)

        # (B) 배치 기준 리트리벌 후보 추출
        obs_b, _, _, next_obs_b, _ = batch
        if retriever is not None and config.retrieval.enabled:
            # 현재 상태용 후보
            model_eval = _build_eval_model(train_state)  # 가볍게 최신 파라미터 반영
            retrieval_now = retriever.query_actions(model_eval, obs_b)  # [B,R,H,A] (또는 None)
            # 타깃(critic target)용 후보
            if getattr(config.retrieval, "use_for_targets", True):
                retrieval_next = retriever.query_actions(model_eval, next_obs_b)
            else:
                retrieval_next = None
        else:
            retrieval_now = None
            retrieval_next = None


        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch, retrieval_now, retrieval_next)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

            wandb.log(
                {
                    'learning_phase': 0 if step < config.offline_steps else 1,
                    'replay/num_trajs': replay_buffer.size,
                    'replay/num_steps': replay_buffer.total_steps,
                    'is_success (exploration)': int(traj['is_success']),
                },
                step=step
            )
            
        if step % config.eval_interval == 0:
            eval_rng = jax.random.fold_in(eval_rng, step)
            eval_rng, task_rng = jax.random.split(eval_rng)
            task_id = jax.random.randint(task_rng, (), 0, num_tasks)
            task = task_suite.get_task(task_id)
            eval_env, task_description = get_libero_env(task, 256, config.seed)
            logging.info(f"Eval at step {step}: task_id={task_id}, task_description='{task_description}'")

            perform_control_eval(eval_rng, config, train_state, eval_env, task_description, step, tokenizer)


        batch = next(data_iter)
        

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
