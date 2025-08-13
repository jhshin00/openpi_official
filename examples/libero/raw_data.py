import numpy as np
import tensorflow_datasets as tfds


def inspect_libero_dataset(raw_name: str, data_dir: str = None):
    """
    raw_name: e.g. "libero_10_no_noops"
    data_dir: TFDS가 저장된 디렉토리 (없으면 기본 경로 사용)
    """
    ds = tfds.load(raw_name, split="train", data_dir=data_dir, shuffle_files=False)

    # 첫 번째 episode
    for episode in tfds.as_numpy(ds.take(1)):
        print("== Episode keys ==")
        print(list(episode.keys()))

        # steps 안의 첫 스텝
        steps = episode["steps"]
        first_step = next(iter(steps))
        print("\n== Keys inside episode['steps'] ==")
        print(list(first_step.keys()))

        # 각 필드 타입/크기 보기
        print("\n== first_step[field] info ==")
        for k, v in first_step.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: array shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (bytes, str)):
                # bytes면 길이, str이면 길이
                print(f"  {k}: {type(v).__name__}, length={len(v)}")
            elif isinstance(v, (bool, np.bool_)):
                print(f"  {k}: bool, value={v}")
            elif np.isscalar(v):
                print(f"  {k}: scalar {type(v).__name__}, value={v}")
            else:
                # 혹시 nested dict나 리스트
                try:
                    print(f"  {k}: {type(v)}, len={len(v)}")
                except Exception:
                    print(f"  {k}: {type(v)}")
        break


if __name__ == "__main__":
    inspect_libero_dataset("libero_10_no_noops", data_dir="datasets/libero_raw")
