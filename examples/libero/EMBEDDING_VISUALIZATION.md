# VLM Embedding Visualization for LIBERO

이 가이드는 fine-tuned된 PI0 policy를 LIBERO 환경에서 실행하면서 VLM (Vision-Language Model) embedding을 수집하고 시각화하는 방법을 설명합니다.

## 개요

이 기능을 사용하면 다음을 수행할 수 있습니다:
1. LIBERO 환경에서 policy를 실행하면서 각 step의 VLM embedding 수집
2. Task별로, success/failure별로 embedding 데이터 저장
3. t-SNE나 UMAP을 사용하여 embedding space를 2D로 시각화
4. Task별로 embedding이 어떻게 클러스터링되는지 분석

## 사용 방법

### 1단계: Embedding 수집

먼저 LIBERO 환경에서 policy를 실행하면서 embedding을 수집합니다.

#### 방법 1: Local Policy 사용 (권장)

```bash
python examples/libero/main.py \
    --use_local_policy \
    --checkpoint_config pi0_libero \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_libero \
    --save_embeddings \
    --embeddings_out_path data/libero/embeddings \
    --task_suite_name libero_spatial \
    --num_trials_per_task 3
```

#### 방법 2: Websocket Server 사용 (embedding 수집 불가)

Websocket 방식은 현재 embedding 추출을 지원하지 않습니다. Embedding 수집을 위해서는 `--use_local_policy` 옵션을 사용해야 합니다.

### 주요 파라미터 설명

- `--use_local_policy`: Local mode로 policy를 직접 로드 (embedding 수집에 필수)
- `--checkpoint_config`: Training config 이름 (예: `pi0_libero`)
- `--checkpoint_dir`: Checkpoint 경로 (예: `gs://openpi-assets/checkpoints/pi0_libero`)
- `--save_embeddings`: Embedding 수집 활성화
- `--embeddings_out_path`: Embedding 저장 경로
- `--task_suite_name`: LIBERO task suite 선택
  - `libero_spatial`: 10 tasks, max 220 steps
  - `libero_object`: 10 tasks, max 280 steps
  - `libero_goal`: 10 tasks, max 300 steps
  - `libero_10`: 10 tasks, max 520 steps
  - `libero_90`: 90 tasks, max 400 steps
- `--num_trials_per_task`: 각 task당 실행 횟수 (기본값: 3)

### 수집되는 데이터

실행이 완료되면 `embeddings_out_path`에 다음과 같은 pickle 파일이 저장됩니다:

```
data/libero/embeddings/embeddings_libero_spatial.pkl
```

이 파일에는 다음 정보가 포함됩니다:
```python
[
    {
        "metadata": {
            "task_id": 0,
            "task_description": "put the black bowl on top of the cabinet",
            "episode_idx": 0,
            "success": True,
            "num_timesteps": 150,
        },
        "embeddings": [
            {
                "timestep": 10,
                "embedding": np.array([...]),  # (d_emb,) VLM embedding vector
            },
            # ... more timesteps
        ],
    },
    # ... more episodes
]
```

### 2단계: Embedding 시각화

수집된 embedding을 t-SNE와 UMAP으로 시각화합니다.

```bash
python examples/libero/visualize_embeddings.py \
    --embeddings_path data/libero/embeddings/embeddings_libero_spatial.pkl \
    --output_dir data/libero/visualizations
```

#### 고급 옵션

```bash
python examples/libero/visualize_embeddings.py \
    --embeddings_path data/libero/embeddings/embeddings_libero_spatial.pkl \
    --output_dir data/libero/visualizations \
    --tsne_perplexity 30 \
    --tsne_n_iter 1000 \
    --umap_n_neighbors 15 \
    --umap_min_dist 0.1
```

시각화 파라미터:
- `--tsne_perplexity`: t-SNE perplexity (기본값: 30, 범위: 5-50)
- `--tsne_n_iter`: t-SNE 반복 횟수 (기본값: 1000)
- `--umap_n_neighbors`: UMAP 이웃 개수 (기본값: 15)
- `--umap_min_dist`: UMAP 최소 거리 (기본값: 0.1)
- `--skip_tsne`: t-SNE 시각화 건너뛰기
- `--skip_umap`: UMAP 시각화 건너뛰기

### 생성되는 시각화

다음 이미지 파일들이 생성됩니다:

1. **`tsne_visualization.png`**: t-SNE 2D projection
   - 왼쪽: Task별로 색상 구분
   - 오른쪽: Success(초록)/Failure(빨강) 구분

2. **`tsne_by_task.png`**: Task별 개별 시각화 (task 수 ≤ 10일 때)
   - 각 subplot에서 해당 task만 강조 표시

3. **`umap_visualization.png`**: UMAP 2D projection
   - 왼쪽: Task별로 색상 구분
   - 오른쪽: Success(초록)/Failure(빨강) 구분

4. **`umap_by_task.png`**: Task별 개별 시각화 (task 수 ≤ 10일 때)
   - 각 subplot에서 해당 task만 강조 표시

## 분석 예시

### Task Clustering 분석
- 같은 task의 embedding이 가까이 클러스터링되는지 확인
- 유사한 task들이 embedding space에서 가까운지 확인

### Success vs Failure 분석
- 성공한 trajectory와 실패한 trajectory의 embedding 분포 비교
- Failure case가 특정 영역에 몰려있는지 확인

### Trajectory 진행 분석
- 같은 episode 내에서 timestep에 따라 embedding이 어떻게 변화하는지 추적
- Task 진행에 따른 embedding space 이동 패턴 분석

## 요구사항

### Python 패키지

기본 요구사항 (이미 설치되어 있음):
```bash
numpy
matplotlib
scikit-learn
seaborn
```

선택적 요구사항 (UMAP 시각화를 위해):
```bash
pip install umap-learn
```

## 비교 시각화 (Comparison Visualization)

두 개의 embedding 파일을 비교하여 다음을 분석할 수 있습니다:
- **다른 모델 비교**: Base model vs Finetuned model
- **다른 Task Suite 비교**: libero_spatial vs libero_goal
- **학습 진행 비교**: Early checkpoint vs Final checkpoint

### 비교 시각화 실행

#### Python 스크립트 사용 (2개 비교)
```bash
python examples/libero/compare_embeddings.py \
    --embeddings_paths data/libero/embeddings/libero_finetune/embeddings_libero_spatial.pkl \
                      data/libero/embeddings/libero_finetune/embeddings_libero_goal.pkl \
    --labels "Spatial Suite (Finetuned)" "Goal Suite (Finetuned)" \
    --output_dir data/libero/comparisons
```

#### 여러 개 비교 (3개 이상)
```bash
python examples/libero/compare_embeddings.py \
    --embeddings_paths data/libero/embeddings/libero_finetune/embeddings_libero_spatial.pkl \
                      data/libero/embeddings/libero_finetune/embeddings_libero_goal.pkl \
                      data/libero/embeddings/libero_finetune/embeddings_libero_object.pkl \
                      data/libero/embeddings/libero_no_finetune/embeddings_libero_goal.pkl \
    --labels "Spatial" "Goal" "Object" "Goal_X" \
    --output_dir data/libero/comparisons/4
```

### 생성되는 비교 시각화

1. **`comparison_tsne.png`**: t-SNE 비교 시각화
   - 첫 번째 subplot: 모든 embedding set을 함께 표시 (색상/마커로 구분)
   - 나머지 subplots: 각 set을 개별적으로 강조 표시
   - 마커: ○ (set 1), △ (set 2), □ (set 3), ◇ (set 4), ...

2. **`comparison_tsne_by_task.png`**: Task별 비교
   - 색상 = task ID (rainbow colormap)
   - 마커 = embedding source
   - 모든 set의 task 분포를 한 눈에 비교

3. **`comparison_umap.png`**: UMAP 비교 시각화
   - t-SNE와 동일한 구조
   - UMAP의 글로벌 구조 보존 특성으로 다른 관점 제공

### 비교 분석 예시

**여러 Task Suite 비교**:
```bash
python examples/libero/compare_embeddings.py \
    --embeddings_paths data/libero/embeddings/embeddings_libero_spatial.pkl \
                      data/libero/embeddings/embeddings_libero_goal.pkl \
                      data/libero/embeddings/embeddings_libero_object.pkl \
    --labels "Spatial Suite" "Goal Suite" "Object Suite"
```

**Base vs Finetuned 비교**:
```bash
python examples/libero/compare_embeddings.py \
    --embeddings_paths data/libero/embeddings/base_model.pkl \
                      data/libero/embeddings/finetuned_model.pkl \
    --labels "Base Model" "Finetuned Model"
```

분석 가능한 인사이트:
- 서로 다른 task suite의 embedding이 어떻게 분포하는가?
- Finetuned model의 embedding이 더 클러스터링되어 있는가?
- Task 구분이 더 명확해졌는가?
- Embedding space가 어떻게 변화했는가?
- 여러 suite/model을 동시에 비교하여 패턴 발견

## 예시 실행

### 전체 파이프라인 실행

```bash
# 1. Embedding 수집 (libero_spatial, 3 trials per task)
python examples/libero/main.py \
    --use_local_policy \
    --checkpoint_config pi0_libero \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_libero \
    --save_embeddings \
    --task_suite_name libero_spatial \
    --num_trials_per_task 3

# 2. 시각화 생성
python examples/libero/visualize_embeddings.py \
    --embeddings_path data/libero/embeddings/embeddings_libero_spatial.pkl

# 생성된 이미지 확인
ls data/libero/embeddings/*.png
```

### 다른 Task Suite 실행

```bash
# libero_object suite
python examples/libero/main.py \
    --use_local_policy \
    --checkpoint_config pi0_libero \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi0_libero \
    --save_embeddings \
    --task_suite_name libero_object \
    --num_trials_per_task 5

python examples/libero/visualize_embeddings.py \
    --embeddings_path data/libero/embeddings/embeddings_libero_object.pkl
```

## 트러블슈팅

### 문제: "checkpoint_config and checkpoint_dir must be provided"
해결: `--use_local_policy` 사용 시 두 파라미터 모두 제공해야 합니다.

### 문제: "vlm_embedding not in result"
해결: `--use_local_policy` 옵션이 활성화되어 있고, `--save_embeddings`가 설정되어 있는지 확인하세요.

### 문제: "UMAP not available"
해결: `pip install umap-learn`로 UMAP을 설치하거나 `--skip_umap` 옵션을 사용하세요.

### 문제: Memory 부족
해결: `--num_trials_per_task`를 줄이거나, task suite를 작은 것으로 선택하세요.

## 기술적 세부사항

### Embedding 추출 방식
- PI0 모델의 `embed_prefix()` 메서드를 사용하여 VLM embedding 추출
- Image tokens + Language tokens를 모두 포함
- Sequence dimension에 대해 average pooling 적용
- Valid tokens만 사용 (mask 적용)

### Embedding 수집 주기
- **매 step마다 embedding 수집** (action inference와 독립적)
- `extract_embedding()` 메서드를 사용하여 action inference 없이 embedding만 추출
- 이를 통해 모든 timestep에서의 VLM embedding 변화를 추적 가능
- Action inference는 여전히 `replan_steps` (기본값: 5)마다만 수행하여 효율성 유지

### 파일 구조
```
examples/libero/
├── main.py                      # 메인 실행 스크립트 (embedding 수집 기능 추가)
├── visualize_embeddings.py      # 시각화 스크립트
└── EMBEDDING_VISUALIZATION.md   # 이 문서

src/openpi/policies/
├── embedding_policy.py          # VLM embedding을 반환하는 Policy 클래스
└── policy_config.py             # create_trained_embedding_policy 함수 추가
```

## 참고

- PI0 모델 논문: [Physical Intelligence](https://www.physicalintelligence.company/)
- LIBERO 벤치마크: [LIBERO](https://libero-project.github.io/)
- t-SNE: [scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- UMAP: [UMAP Documentation](https://umap-learn.readthedocs.io/)

