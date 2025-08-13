from src.openpi.training.config import get_config
import jax
import jax.numpy as jnp
import src.openpi.training.data_loader as _data_loader
import src.openpi.training.checkpoints as _checkpoints
import src.openpi.training.sharding as sharding
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

config = get_config("pi0_fql_libero_lora_finetune")
if config.batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size {config.batch_size} must be divisible by number of devices.")

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

data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
data_iter = iter(data_loader)

# Get the first batch to access is_pad information
obs, acts, rews, terminal, next_obs, actions_is_pad, reward_is_pad = next(data_iter)

# 1. observation (dict) 안에 어떤 키가 들어 있는지
print("obs keys      :", obs.__dict__.keys())              # ex) dict_keys(['state', 'image', 'wrist_image', ...])
print("next_obs keys :", next_obs.__dict__.keys())

# 2. 핵심 텐서 shape 확인
print("\n--- shapes ----------------------------")
print("state        :", obs.state.shape)       # (B,  …)   s_t
print("next_state   :", next_obs.state.shape)  # (B,  …)   s_{t+H}
print("image keys   :", list(obs.images.keys()))  # 이미지 키들 확인
print("first image  :", list(obs.images.values())[0].shape if obs.images else "No images")  # 첫 번째 이미지 shape
print("actions      :", acts.shape)               # (B, H, A)
print("rewards      :", rews.shape)               # (B, H)
print("terminal     :", terminal.shape)           # (B,)

# 3. 첫 샘플을 눈으로 보기  (jax.Array → numpy 변환)
i = 0
print("\n--- sample 0 ---------------------------")
print("s_t          :", jax.device_get(obs.state[i])[:8])         # 앞 8차원만
print("s_t+H        :", jax.device_get(next_obs.state[i])[:8])
print("reward seq   :", jax.device_get(rews[i]))                     # 길이 H
print("a_t          :", jax.device_get(acts[i, 0]))                  # a_t
print("a_t+H-1      :", jax.device_get(acts[i, -1]))                 # a_{t+H-1}
print("terminal flag:", int(terminal[i]))

print("obs.images.shape      :", list(obs.images.values())[0].shape if obs.images else "No images")        # (B, C, H, W) 또는 (B, H, W, C)
print("next_obs.images.shape :", list(next_obs.images.values())[0].shape if next_obs.images else "No images")

# 2) 시각화 ----------------------------------------
#   └─ 첫 번째 샘플(i=0)만 예시
n_samples = 10                    # 보고 싶은 샘플 수 (배치보다 작게)
H = acts.shape[1]

# 시각화할 sample들의 terminal과 reward 값 출력
print(f"\n--- Visualization Samples ({n_samples}개) ---------------------------")
for i in range(n_samples):
    print(f"Sample {i}:")
    print(f"  Terminal: {int(terminal[i])}")
    print(f"  Reward sequence: {jax.device_get(rews[i])}")
    print(f"  Total reward: {jax.device_get(rews[i]).sum():.3f}")
    print(f"  Mean reward: {jax.device_get(rews[i]).mean():.3f}")
    print(f"  Actions pad: {jax.device_get(actions_is_pad[i])}")
    print(f"  Reward pad: {jax.device_get(reward_is_pad[i])}")
    print(f"  Action sequence (all timesteps):")
    for t in range(acts.shape[1]):
        action_t = jax.device_get(acts[i, t])
        print(f"    t+{t}: {action_t[:7]}...")  # 앞 7개 값만 출력
    print()

fig, axs = plt.subplots(n_samples, 2, figsize=(6, 3*n_samples))

# 첫 번째 이미지 키 가져오기
first_img_key = list(obs.images.keys())[0] if obs.images else None
print(f"Using image key: {first_img_key}")

for row in range(n_samples):
    # (C,H,W) → (H,W,C) 전치 + host 로 복사
    img_t  = jax.device_get(obs.images[first_img_key][row])
    img_tH = jax.device_get(next_obs.images[first_img_key][row])
    
    print(f"Sample {row}: img_t shape={img_t.shape}, img_tH shape={img_tH.shape}")

    if img_t.ndim == 3 and img_t.shape[0] < 10:   # (C,H,W) 일 때
        img_t  = np.transpose(img_t,  (1,2,0))
        img_tH = np.transpose(img_tH, (1,2,0))

    # [0,1] 정규화
    img_t  = (img_t  - img_t.min())  / (img_t.max()  - img_t.min()  + 1e-8)
    img_tH = (img_tH - img_tH.min()) / (img_tH.max() - img_tH.min() + 1e-8)

    # 그림 배치
    axs[row, 0].imshow(img_t)
    axs[row, 0].set_title(f"sample {row} @ t")

    axs[row, 1].imshow(img_tH)
    axs[row, 1].set_title(f"sample {row} @ t+{H}")

    for col in range(2):
        axs[row, col].axis("off")

plt.tight_layout()
plt.savefig('test_dataset_visualization.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'test_dataset_visualization.png'")
plt.close()  # Close the figure to free memory
    

# for step in range(5):
#     batch = next(data_iter)
#     obs, acts, rews, terminal, next_obs = batch
#     # diff = jnp.float32(rews) - jnp.float32(terminal)
#     # print(diff)
#     # if jnp.any(diff != 0).item():
#     #     print("error!!!!!")

#     print(rews)