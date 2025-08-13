#!/usr/bin/env python
# inspect_action_values.py
"""
Parquet 에피소드에서 action 컬럼을 찾아
각 step의 action 값을 출력하고 |action| > 1인 요소를 검사하는 스크립트
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

def load_first_parquet(path: str) -> str:
    """파일 또는 디렉터리 경로를 받아 첫 번째 .parquet 파일의 경로를 리턴"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        if not files:
            raise FileNotFoundError(f"No .parquet found in directory {path}")
        return files[0]
    elif path.endswith('.parquet'):
        return path
    else:
        raise ValueError(f"Not a parquet file or directory: {path}")

def find_col(df: pd.DataFrame, base: str) -> str:
    """'base', 'bases', 'base_vec' 형태로 컬럼 검색"""
    for cand in (base, f"{base}s", f"{base}_vec"):
        if cand in df.columns:
            return cand
    raise KeyError(f"No column found for base '{base}'. Available: {list(df.columns)}")

def print_action_values(df: pd.DataFrame, action_col: str, head: int):
    """앞 head개 스텝의 action 배열과 |action|>1 여부를 출력"""
    n = head if head > 0 else len(df)
    for i in range(min(n, len(df))):
        raw = df[action_col].iloc[i]
        arr = np.asarray(raw)
        print(f"\n--- Step {i} ---")
        print(f"[{action_col}] shape={arr.shape}, dtype={arr.dtype}")
        print("값:", arr.tolist())
        mask = np.abs(arr) > 1.0
        if np.any(mask):
            idxs = np.argwhere(mask).flatten().tolist()
            vals = arr[mask].tolist()
            print(f"⚠️  |action|>1인 요소 {idxs}: {vals}")
        else:
            print("|action| <= 1 입니다.")

def main():
    parser = argparse.ArgumentParser(description="Parquet action 값 검사기")
    parser.add_argument("path", help="Parquet 파일 경로 또는 디렉터리")
    parser.add_argument("--head", "-k", type=int, default=10,
                        help="검사할 첫 N개 행 수 (기본값: 10; 전체: 0 이하)")
    args = parser.parse_args()

    pq = load_first_parquet(args.path)
    print(f"Loading parquet: {pq}")
    df = pd.read_parquet(pq)

    action_col = find_col(df, "action")
    print(f"Found action column: '{action_col}'")

    print_action_values(df, action_col, args.head)

if __name__ == "__main__":
    main()
