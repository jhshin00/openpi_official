#!/usr/bin/env python
# inspect_episode.py
"""
단일 Libero-style Parquet 에피소드 확인 + 시각화 스크립트
  · action 타입(Δ vs absolute) 판정
  · state / actions / next_state / 이미지 미리보기
  · --show-images 로 PNG/JPEG 실제 렌더링
"""

import os, glob, argparse, io, warnings
import numpy as np, pandas as pd
from PIL import Image

# ──────────────────────────────────────────────────────── 설정
# KEEP_COLS = ["image", "wrist_image", "state", "actions",
#              "next_image", "next_wrist_image", "next_state"]
KEEP_COLS = [ "state", "actions",
             "next_state"]

# ──────────────────────────────────────────────────────── 유틸
def _find(df, base):
    for cand in (base, f"{base}s", f"{base}_vec"):
        if cand in df.columns:
            return cand
    raise KeyError(f"❌ {base}(s) 컬럼 없음. 실제: {list(df.columns)}")

def _fmt_obj(val, max_len=500):
    if isinstance(val, (bytes, bytearray)):
        if val[:8] == b"\x89PNG\r\n\x1a\n":
            return f"(PNG {len(val)} B)"
        if val[:2] == b"\xff\xd8":
            return f"(JPEG {len(val)} B)"
        return f"(bytes {len(val)} B)"
    if isinstance(val, dict) and "bytes" in val:             # ★
        return _fmt_obj(val["bytes"], max_len)               # ★
    txt = str(val)
    return txt if len(txt) <= max_len else txt[:max_len] + "…"

def _summarize(df, rows=10):
    df2 = df.copy()[[c for c in KEEP_COLS if c in df.columns]]
    for col in df2.select_dtypes("object"):
        df2[col] = df2[col].apply(_fmt_obj)
    return df2.head(rows).to_string(index=False)

# ──────────────────────────────────────────────────────── 핵심
def classify(df, tol=1e-3):
    s_col, a_col, ns_col = (_find(df, k) for k in ("state", "action", "next_state"))
    s  = np.stack(df[s_col].to_numpy())
    a  = np.stack(df[a_col].to_numpy())
    ns = np.stack(df[ns_col].to_numpy())

    if a.shape[1] == 7:                     # 6D pose + gripper
        s, a, ns = s[:, :6], a[:, :6], ns[:, :6]

    mae_d = np.mean(np.abs((ns - s) - a))
    mae_a = np.mean(np.abs(ns - a))
    tag   = "delta" if mae_d + tol < mae_a else "absolute"
    return tag, mae_d, mae_a, (s_col, a_col, ns_col)

def show_images(df, n_imgs=3):
    import matplotlib.pyplot as plt
    imgs = []
    for _, row in df.head(n_imgs).iterrows():
        for key in ("image", "wrist_image"):
            if key not in row:
                continue
            blob = row[key]
            if isinstance(blob, dict) and "bytes" in blob:   # ★ dict wrapping 처리
                blob = blob["bytes"]
            if not isinstance(blob, (bytes, bytearray)):
                continue
            imgs.append((key, blob))
    if not imgs:
        print("⚠️  표시할 이미지 바이트 컬럼이 없습니다.")
        return
    for idx, (name, b) in enumerate(imgs, 1):
        try:
            img = Image.open(io.BytesIO(b))
        except Exception as e:
            warnings.warn(f"{name} 디코딩 실패: {e}")
            continue
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{idx}: {name}")
    plt.show()

def load_first_parquet(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError("디렉터리에 *.parquet 없음")
        return files[0]
    return path

# ──────────────────────────────────────────────────────── CLI
def main():
    ap = argparse.ArgumentParser("Parquet 에피소드 검사기")
    ap.add_argument("path", help="Parquet 파일 또는 폴더")
    ap.add_argument("--print-all", action="store_true",
                    help="DataFrame 전체 출력")
    ap.add_argument("--head", "-k", type=int, default=10,
                    help="미리보기 행 수 (기본 10)")
    ap.add_argument("--show-images", action="store_true",
                    help="이미지 렌더링")
    ap.add_argument("-n", "--num-imgs", type=int, default=3,
                    help="--show-images 시 보여줄 이미지 수")
    args = ap.parse_args()

    p = load_first_parquet(args.path)
    df = pd.read_parquet(p)

    tag, mae_d, mae_a, cols = classify(df)
    print(f"\n{os.path.basename(p)}  ⇒  {tag.upper():8s}"
          f" | MAE_Δ={mae_d:.4e}  MAE_abs={mae_a:.4e}")
    print(f"사용 컬럼: state='{cols[0]}', action='{cols[1]}', next_state='{cols[2]}'\n")

    print(_summarize(df, None if args.print_all else args.head))
    if not args.print_all and len(df) > args.head:
        print(f"... (앞 {args.head}행; --print-all 로 전체)")
        
    # ────────────────────────────────────────────────────────
    # 각 step의 reward & terminal 출력
    def _find_col(base):
        for cand in (base, f"{base}s", f"{base}_vec"):
            if cand in df.columns:
                return cand
        return None

    reward_col = _find_col("reward")
    term_col   = _find_col("terminal")
    if reward_col or term_col:
        print("\n--- reward & terminal per step ---")
        for i, row in df.iterrows():
            r = row[reward_col] if reward_col else "N/A"
            t = row[term_col]   if term_col   else "N/A"
            print(f"step {i:4d} | reward={r} | terminal={t}")


    if args.show_images:
        show_images(df, args.num_imgs)

if __name__ == "__main__":
    main()




# import pandas as pd, numpy as np

# df = pd.read_parquet("./datasets/libero_FQL/data/chunk-000/episode_000154.parquet")
# s   = np.stack(df["state"])
# print("corr(state[:,6], state[:,7]) =", np.corrcoef(s[:,6], s[:,7])[0,1])
# print("unique values 7th dim (first 20)", np.unique(np.round(s[:2000,7],4)))