"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_expo as pi0_expo
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

@dataclass
class BufferConfig:
    capacity_total: int = 100_000         # 총 step 수 기준
    batch_offline_ratio: float = 0.5      # 배치 내 offline:online 비율
    success_memory_per_task: int = 10     # 성공 메모리 크기
    eviction: str = "fifo"                # online eviction 정책

@dataclass
class RetrievalConfig:
    enabled: bool = True
    topk: int = 8                         # 한 배치당 후보 traj/action 묶음 개수
    success_only: bool = True             # 성공 traj만 인덱싱
    alpha_q: float = 0.3                  # Q 혼합 가중 등(너의 SimpleRetriever 정의에 맞춤)
    refresh_every: int = 2000             # 스텝 단위 인덱스 리프레시
    use_for_targets: bool = True          # critic target(next action)에도 사용
    use_for_rollout: bool = True          # 환경 상호작용 시에도 사용



@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "expo_project"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule_actor: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    lr_schedule_critic: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    lr_schedule_temp: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    lr_schedule_edit_actor: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer_actor: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    optimizer_edit_actor: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    optimizer_critic: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    optimizer_temp: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    tau_target: float = 0.005

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"
    # Base directory for experiments (videos, logs, etc.)
    experiments_dir: str = "./experiments"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1
    
    # Expo buffer specific settings
    use_offline_data: bool = True
    libero_data_dir: str = "/ssd2/EXPO/datasets/libero_goal"
    offline_dataset_subset_num: int | None = None
    capacity: int = 100000  # Buffer capacity in steps
    action_horizon: int = 50
    action_dim: int = 32
    max_token_len: int = 48
    
    # Offline-Online learning settings
    offline_steps: int = 5000  # Number of steps for offline learning
    libero_task_suite: str = "libero_goal"  # Libero task suite to use
    env_reuse_frequency: int = 10  # Reuse environment for N episodes
    start_updates: int = 1000  # Start training after this many samples
    num_update_steps: int = 1  # Number of update steps per environment step
    eval_interval: int = 1000  # Evaluation interval
    eval_episodes: int = 1  # Number of episodes for evaluation
    max_timesteps: int = 200  # Maximum timesteps per episode
    env_max_reward: float = 1.0  # Maximum reward in environment
    num_steps_wait: int = 0  # Steps to wait before starting action
    rollout_interval: int = 100  # Rollout interval

    buffer: BufferConfig = BufferConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi0_expo_libero",
        exp_name="exp",
        model=pi0_expo.Pi0ExpoConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            encoder_sharing=False,
            img_latent_dim=256,
            txt_latent_dim=256,
            state_dim=128,
            hidden_dim=256,
            out_dim=256,
            vocab_size=257_152,
            image_keys=["base_0_rgb", "left_wrist_0_rgb"],
            d_model=256,
            n_layers=2,
            kernel_size=3,
            dropout_rate=0.1,
            use_layer_norm=True,
            use_film_gate=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=48,
            discount=0.99,
            n_base_samples=2,
            n_edit_samples=2,
            entropy_scale=1.0,
            target_entropy=-32.0,
            edit_action_scale=0.05,
            initial_temperature=1.0,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=50_000,
        freeze_filter=pi0_expo.Pi0ExpoConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            encoder_sharing=False,
            img_latent_dim=256,
            txt_latent_dim=256,
            state_dim=128,
            hidden_dim=256,
            out_dim=256,
            vocab_size=257_152,
            image_keys=["base_0_rgb", "left_wrist_0_rgb"],
            d_model=256,
            n_layers=2,
            kernel_size=3,
            dropout_rate=0.1,
            use_layer_norm=True,
            use_film_gate=True,
            action_dim=32,
            action_horizon=50,
            max_token_len=48,
            discount=0.99,
            n_base_samples=2,
            n_edit_samples=2,
            entropy_scale=1.0,
            target_entropy=-32.0,
            edit_action_scale=0.05,
            initial_temperature=1.0,
        ).get_freeze_filter(),
        ema_decay=None,

        log_interval=10,
        save_interval=2500,
        keep_period=5000,
        overwrite=True,
        resume=False,

        wandb_enabled=True,

        batch_size=32,
        tau_target=0.005,

        lr_schedule_actor = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 2.5e-5,
            decay_steps = 30_000,
            decay_lr = 2.5e-6,
        ),
        lr_schedule_critic = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 1.0e-4,
            decay_steps = 30_000,
            decay_lr = 1.0e-5,
        ),
        lr_schedule_edit_actor = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5.0e-4,
            decay_steps = 30_000,
            decay_lr = 5.0e-5,
        ),
        lr_schedule_temp = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5.0e-4,
            decay_steps = 30_000,
            decay_lr = 5.0e-5,
        ),
        optimizer_actor = _optimizer.AdamW(),
        optimizer_critic = _optimizer.AdamW(),
        optimizer_edit_actor = _optimizer.AdamW(),
        optimizer_temp = _optimizer.AdamW(),

        seed = 42,
    ),

]



if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
