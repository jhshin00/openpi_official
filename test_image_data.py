#!/usr/bin/env python
# inspect_image_values.py
"""
Parquet 에피소드에서 이미지 컬럼을 찾아
픽셀 값을 NumPy 배열로 출력하는 스크립트
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import io

def load_first_parquet(path: str) -> str:
    """파일 혹은 디렉터리 경로를 받아 첫 번째 .parquet 파일의 경로를 리턴"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.parquet')))
        if not files:
            raise FileNotFoundError(f"No .parquet found in directory {path}")
        return files[0]
    elif path.endswith('.parquet'):
        return path
    else:
        raise ValueError(f"Not a parquet file or directory: {path}")

def find_image_columns(df: pd.DataFrame) -> list[str]:
    """DataFrame에서 bytes 혹은 dict(bytes=...) 타입 컬럼을 찾아 이름 리턴"""
    img_cols = []
    for col in df.columns:
        sample = df[col].iloc[0]
        if isinstance(sample, (bytes, bytearray)) or \
           (isinstance(sample, dict) and 'bytes' in sample):
            img_cols.append(col)
    return img_cols

def decode_image(blob: bytes) -> np.ndarray:
    """바이트 스트림을 PIL로 열어 NumPy 배열로 변환"""
    with io.BytesIO(blob) as bio:
        img = Image.open(bio)
        arr = np.array(img)
    return arr

def print_image_values(df: pd.DataFrame, cols: list[str], head: int):
    n = min(head, len(df))
    for i in range(n):
        print(f"\n--- Row {i} ---")
        for col in cols:
            raw = df[col].iloc[i]
            # dict wrapping 처리
            blob = raw['bytes'] if isinstance(raw, dict) and 'bytes' in raw else raw
            if not isinstance(blob, (bytes, bytearray)):
                print(f"[{col}] not bytes (type={type(raw)})")
                continue
            try:
                arr = decode_image(blob)
            except Exception as e:
                print(f"[{col}] decode 실패: {e}")
                continue
            print(f"[{col}] shape={arr.shape}, dtype={arr.dtype}")
            # 전체 픽셀값 출력 (주의: 큰 배열은 터미널이 느려집니다)
            print(arr)
            
def main():
    p = argparse.ArgumentParser(description="Parquet 이미지 픽셀값 출력기")
    p.add_argument("path", help="Parquet 파일 경로 또는 디렉터리")
    p.add_argument("--head", "-k", type=int, default=3,
                   help="출력할 행 수 (기본값: 3)")
    args = p.parse_args()

    pq = load_first_parquet(args.path)
    print(f"Loading parquet: {pq}")
    df = pd.read_parquet(pq)

    img_cols = find_image_columns(df)
    if not img_cols:
        print("⚠️ 이미지 바이트 컬럼을 찾을 수 없습니다.")
        return
    print(f"Found image columns: {img_cols}")

    print_image_values(df, img_cols, args.head)

if __name__ == "__main__":
    main()
