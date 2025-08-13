#!/usr/bin/env python
"""
UR3 데이터셋의 action이 delta action인지 absolute action인지 확인하는 개선된 스크립트

inspect_episode.py의 classify 함수 기반:
- MAE_delta = mean(|(next_state - state) - action|)
- MAE_absolute = mean(|next_state - action|)
- MAE_delta < MAE_absolute 이면 delta action
- MAE_delta >= MAE_absolute 이면 absolute action

UR3 로봇 구조:
- 6개 joint (shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
- 1개 gripper (마지막 7번째 값)
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns

def classify_action_type(actions: np.ndarray, joint_positions: np.ndarray, tol: float = 1e-3) -> Tuple[str, float, float]:
    """
    Action 타입을 판단하는 함수
    
    Args:
        actions: (n_frames, 7) - 6개 joint + 1개 gripper
        joint_positions: (n_frames, 7) - 6개 joint + 1개 gripper
        tol: 허용 오차
    
    Returns:
        action_type: "delta" 또는 "absolute"
        mae_delta: delta action 가정 시 MAE
        mae_absolute: absolute action 가정 시 MAE
    """
    
    # 6개 joint만 사용 (gripper 제외)
    s = joint_positions[:, :6]   # state (n_frames, 6)
    a = actions[:, :6]           # actions (n_frames, 6)
    ns = joint_positions[:, :6]  # next_state (n_frames, 6)
    
    # Delta action 가정: action = next_state - state
    # MAE_delta = mean(|(next_state - state) - action|)
    mae_delta = np.mean(np.abs((ns - s) - a))
    
    # Absolute action 가정: next_state = action
    # MAE_absolute = mean(|next_state - action|)
    mae_absolute = np.mean(np.abs(ns - a))
    
    # 판단: MAE_delta + tol < MAE_absolute 이면 delta action
    action_type = "delta" if mae_delta + tol < mae_absolute else "absolute"
    
    return action_type, mae_delta, mae_absolute

def analyze_hdf5_file(hdf5_path: Path) -> Dict:
    """HDF5 파일을 분석하여 action 타입과 통계 정보 반환"""
    
    print(f"🔍 분석 중: {hdf5_path.name}")
    
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        
        # 데이터 추출
        actions = data_group["actions"][:]  # (n_frames, 7)
        joint_positions = data_group["joint_positions"][:]  # (n_frames, 7)
        
        n_frames = actions.shape[0]
        
        # Action 타입 판단
        action_type, mae_delta, mae_absolute = classify_action_type(actions, joint_positions)
        
        # 각 joint별로도 분석
        joint_analysis = []
        for i in range(6):  # 6개 joint에 대해
            s_joint = joint_positions[:, i:i+1]  # (n_frames, 1)
            a_joint = actions[:, i:i+1]          # (n_frames, 1)
            ns_joint = joint_positions[:, i:i+1] # (n_frames, 1)
            
            mae_d_joint = np.mean(np.abs((ns_joint - s_joint) - a_joint))
            mae_a_joint = np.mean(np.mean(np.abs(ns_joint - a_joint)))
            
            joint_type = "delta" if mae_d_joint + 1e-3 < mae_a_joint else "absolute"
            
            joint_analysis.append({
                "joint_id": i,
                "joint_name": f"Joint_{i+1}",
                "type": joint_type,
                "mae_delta": mae_d_joint,
                "mae_absolute": mae_a_joint,
                "confidence": abs(mae_d_joint - mae_a_joint) / max(mae_d_joint, mae_a_joint, 1e-6)
            })
        
        # 통계 정보
        action_stats = {
            "mean": np.mean(actions[:, :6], axis=0),  # gripper 제외
            "std": np.std(actions[:, :6], axis=0),
            "min": np.min(actions[:, :6], axis=0),
            "max": np.max(actions[:, :6], axis=0),
        }
        
        return {
            "file_path": str(hdf5_path),
            "n_frames": n_frames,
            "overall_type": action_type,
            "mae_delta": mae_delta,
            "mae_absolute": mae_absolute,
            "confidence": abs(mae_delta - mae_absolute) / max(mae_delta, mae_absolute, 1e-6),
            "joint_analysis": joint_analysis,
            "action_stats": action_stats,
        }

def print_summary_analysis(results: List[Dict]):
    """분석 결과 요약 출력"""
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    print("\n" + "="*100)
    print("📊 UR3 Action Type Analysis Summary (Improved Method)")
    print("="*100)
    
    # 전체 통계
    total_frames = 0
    delta_count = 0
    absolute_count = 0
    all_mae_delta = []
    all_mae_absolute = []
    all_confidences = []
    
    for result in results:
        total_frames += result["n_frames"]
        if result["overall_type"] == "delta":
            delta_count += 1
        else:
            absolute_count += 1
        
        all_mae_delta.append(result["mae_delta"])
        all_mae_absolute.append(result["mae_absolute"])
        all_confidences.append(result["confidence"])
    
    print(f"📈 전체 파일 수: {len(results)}")
    print(f"📈 전체 프레임 수: {total_frames:,}")
    print(f"📊 Action Type 분포:")
    print(f"   - Delta Action: {delta_count}개 ({delta_count/len(results)*100:.1f}%)")
    print(f"   - Absolute Action: {absolute_count}개 ({absolute_count/len(results)*100:.1f}%)")
    
    if all_mae_delta and all_mae_absolute:
        print(f"\n📊 MAE 통계:")
        print(f"   - MAE Delta 평균: {np.mean(all_mae_delta):.6f}")
        print(f"   - MAE Absolute 평균: {np.mean(all_mae_absolute):.6f}")
        print(f"   - 평균 신뢰도: {np.mean(all_confidences):.4f}")
    
    # 각 파일별 상세 정보
    print(f"\n📁 파일별 상세 분석 (처음 10개):")
    for i, result in enumerate(results[:10]):
        print(f"\n   {i+1}. {Path(result['file_path']).name}")
        print(f"      프레임 수: {result['n_frames']}")
        print(f"      Action Type: {result['overall_type'].upper()}")
        print(f"      MAE Delta: {result['mae_delta']:.6f}")
        print(f"      MAE Absolute: {result['mae_absolute']:.6f}")
        print(f"      신뢰도: {result['confidence']:.4f}")
    
    if len(results) > 10:
        print(f"   ... 및 {len(results)-10}개 파일 더")

def plot_analysis_results(results: List[Dict], output_dir: Path = None):
    """분석 결과를 시각화"""
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    # 데이터 준비
    types = [r["overall_type"] for r in results]
    mae_deltas = [r["mae_delta"] for r in results]
    mae_absolutes = [r["mae_absolute"] for r in results]
    confidences = [r["confidence"] for r in results]
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("UR3 Action Type Analysis (Improved Method)", fontsize=16)
    
    # 1. Action Type 분포
    type_counts = pd.Series(types).value_counts()
    axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title("Action Type Distribution")
    
    # 2. MAE 비교 (Delta vs Absolute)
    axes[0, 1].scatter(mae_deltas, mae_absolutes, alpha=0.6, s=20)
    axes[0, 1].plot([0, max(max(mae_deltas), max(mae_absolutes))], 
                     [0, max(max(mae_deltas), max(mae_absolutes))], 'r--', alpha=0.7)
    axes[0, 1].set_xlabel("MAE Delta")
    axes[0, 1].set_ylabel("MAE Absolute")
    axes[0, 1].set_title("MAE Delta vs Absolute")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 신뢰도 분포
    axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.3f}')
    axes[1, 0].set_xlabel("Confidence")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Classification Confidence Distribution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Joint별 분석 (첫 번째 파일 기준)
    if results:
        first_result = results[0]
        joint_names = [j["joint_name"] for j in first_result["joint_analysis"]]
        joint_types = [j["type"] for j in first_result["joint_analysis"]]
        
        type_colors = ["green" if t == "delta" else "red" for t in joint_types]
        axes[1, 1].bar(joint_names, [1]*len(joint_names), color=type_colors, alpha=0.7)
        axes[1, 1].set_title(f"Joint-wise Analysis: {Path(first_result['file_path']).name}")
        axes[1, 1].set_ylabel("Type (Green=Delta, Red=Absolute)")
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "action_analysis_improved.png", dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장됨: {output_dir / 'action_analysis_improved.png'}")
    
    plt.show()

def main():
    """메인 함수"""
    
    # UR3 데이터셋 경로
    dataset_path = Path("/ssd2/openpi/datasets/ur3_datasets")
    
    if not dataset_path.exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {dataset_path}")
        return
    
    # HDF5 파일들 찾기
    hdf5_files = sorted(dataset_path.glob("**/data.hdf5"))
    
    if not hdf5_files:
        print(f"❌ data.hdf5 파일을 찾을 수 없습니다: {dataset_path}")
        return
    
    print(f"🔍 {len(hdf5_files)}개의 data.hdf5 파일을 찾았습니다.")
    
    # 분석할 파일 수 제한 (처리 시간 고려)
    max_files = len(hdf5_files) #min(50, len(hdf5_files))
    print(f"📊 처음 {max_files}개 파일을 분석합니다...")
    
    # 각 파일 분석
    results = []
    for i, hdf5_file in enumerate(hdf5_files[:max_files]):
        try:
            result = analyze_hdf5_file(hdf5_file)
            results.append(result)
            print(f"   ✅ {i+1}/{max_files}: {hdf5_file.name} → {result['overall_type'].upper()}")
        except Exception as e:
            print(f"   ❌ {i+1}/{max_files}: {hdf5_file.name} - 오류: {e}")
    
    if not results:
        print("❌ 분석할 수 있는 파일이 없습니다.")
        return
    
    # 결과 출력
    print_summary_analysis(results)
    
    # 그래프 생성
    try:
        plot_analysis_results(results)
    except Exception as e:
        print(f"⚠️  그래프 생성 중 오류: {e}")
        print("📊 그래프 없이 분석을 완료했습니다.")

if __name__ == "__main__":
    main() 