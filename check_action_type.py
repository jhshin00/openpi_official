"""
UR3 데이터셋의 action이 delta action인지 absolute action인지 확인하는 스크립트

Delta action: 이전 상태에서의 변화량 (상대적 변화)
Absolute action: 절대적인 목표 상태 (절대적 위치/값)

UR3 로봇 구조:
- 6개 joint (shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3)
- 1개 gripper (마지막 7번째 값)

분석 방법:
1. 연속된 action들 간의 차이 계산 (6개 joint만)
2. 차이의 분포와 패턴 분석
3. Joint position과의 관계 분석 (gripper 제외)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import List, Tuple
import pandas as pd

def analyze_action_patterns(hdf5_path: Path) -> dict:
    """HDF5 파일에서 action 패턴을 분석"""
    
    print(f"🔍 분석 중: {hdf5_path}")
    
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        
        # 데이터 추출 (앞 6개는 joint, 마지막 1개는 gripper)
        actions = data_group["actions"][:, :6]  # (n_frames, 6) - gripper 제외
        joint_positions = data_group["joint_positions"][:, :6]  # (n_frames, 6) - gripper 제외
        
        n_frames = actions.shape[0]
        
        # 1. Action 간의 차이 계산 (delta-like behavior)
        action_diffs = np.diff(actions, axis=0)  # (n_frames-1, 6)
        
        # 2. Joint position 간의 차이 계산
        joint_diffs = np.diff(joint_positions, axis=0)  # (n_frames-1, 6)
        
        # 3. Action과 joint position 변화의 상관관계 (6개 joint에 대해)
        correlations = []
        for i in range(6):  # 6개 joint에 대해
            if len(action_diffs) > 0:
                corr = np.corrcoef(action_diffs[:, i], joint_diffs[:, i])[0, 1]
                correlations.append(corr)
            else:
                correlations.append(np.nan)
        
        # 4. 통계 정보
        action_stats = {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "min": np.min(actions, axis=0),
            "max": np.max(actions, axis=0),
        }
        
        action_diff_stats = {
            "mean": np.mean(action_diffs, axis=0) if len(action_diffs) > 0 else np.zeros(6),
            "std": np.std(action_diffs, axis=0) if len(action_diffs) > 0 else np.zeros(6),
            "min": np.min(action_diffs, axis=0) if len(action_diffs) > 0 else np.zeros(6),
            "max": np.max(action_diffs, axis=0) if len(action_diffs) > 0 else np.zeros(6),
        }
        
        return {
            "file_path": str(hdf5_path),
            "n_frames": n_frames,
            "actions": actions,
            "joint_positions": joint_positions,
            "action_diffs": action_diffs,
            "joint_diffs": joint_diffs,
            "correlations": correlations,
            "action_stats": action_stats,
            "action_diff_stats": action_diff_stats,
        }

def plot_action_analysis(results: List[dict], output_dir: Path = None):
    """Action 분석 결과를 시각화"""
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    # 여러 파일의 결과를 종합
    all_correlations = []
    all_action_diffs = []
    all_joint_diffs = []
    
    for result in results:
        all_correlations.extend(result["correlations"])
        if len(result["action_diffs"]) > 0:
            all_action_diffs.append(result["action_diffs"])
            all_joint_diffs.append(result["joint_diffs"])
    
    all_action_diffs = np.vstack(all_action_diffs) if all_action_diffs else np.array([])
    all_joint_diffs = np.vstack(all_joint_diffs) if all_joint_diffs else np.array([])
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("UR3 Action Type Analysis", fontsize=16)
    
    # 1. Action-Joint 상관관계 히스토그램
    if all_correlations:
        axes[0, 0].hist(all_correlations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(all_correlations), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(all_correlations):.3f}')
        axes[0, 0].set_xlabel('Correlation Coefficient')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Action-Joint Position Change Correlation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Action 변화량 분포
    if len(all_action_diffs) > 0:
        for i in range(min(6, all_action_diffs.shape[1])):
            axes[0, 1].hist(all_action_diffs[:, i], bins=30, alpha=0.5, 
                           label=f'Joint {i+1}', density=True)
        axes[0, 1].set_xlabel('Action Difference')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Action Differences Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Joint position 변화량 분포
    if len(all_joint_diffs) > 0:
        for i in range(min(6, all_joint_diffs.shape[1])):
            axes[1, 0].hist(all_joint_diffs[:, i], bins=30, alpha=0.5, 
                           label=f'Joint {i+1}', density=True)
        axes[1, 0].set_xlabel('Joint Position Difference')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Joint Position Differences Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Action vs Joint 변화량 산점도 (첫 번째 joint)
    if len(all_action_diffs) > 0 and len(all_joint_diffs) > 0:
        axes[1, 1].scatter(all_action_diffs[:, 0], all_joint_diffs[:, 0], alpha=0.6, s=1)
        axes[1, 1].set_xlabel('Action Difference (Joint 1)')
        axes[1, 1].set_ylabel('Joint Position Difference (Joint 1)')
        axes[1, 1].set_title('Action vs Joint Position Change (Joint 1)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 상관계수 표시
        if len(all_correlations) > 0:
            corr_joint1 = all_correlations[0]
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_joint1:.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12,
                           bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "action_analysis.png", dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장됨: {output_dir / 'action_analysis.png'}")
    
    plt.show()

def print_summary_analysis(results: List[dict]):
    """분석 결과 요약 출력"""
    
    if not results:
        print("❌ 분석할 결과가 없습니다.")
        return
    
    print("\n" + "="*80)
    print("📊 UR3 Action Type Analysis Summary")
    print("="*80)
    
    # 전체 통계
    all_correlations = []
    total_frames = 0
    
    for result in results:
        all_correlations.extend([c for c in result["correlations"] if not np.isnan(c)])
        total_frames += result["n_frames"]
    
    if all_correlations:
        mean_corr = np.mean(all_correlations)
        std_corr = np.std(all_correlations)
        
        print(f"📈 전체 프레임 수: {total_frames:,}")
        print(f"📊 Action-Joint 상관계수:")
        print(f"   - 평균: {mean_corr:.4f}")
        print(f"   - 표준편차: {std_corr:.4f}")
        print(f"   - 최소값: {np.min(all_correlations):.4f}")
        print(f"   - 최대값: {np.max(all_correlations):.4f}")
        
        # Delta vs Absolute 판단
        print(f"\n🔍 Action Type 판단:")
        if mean_corr > 0.7:
            print(f"   ✅ DELTA ACTION (상관계수: {mean_corr:.4f})")
            print(f"      - Action이 joint position 변화와 높은 상관관계")
            print(f"      - 이전 상태에서의 변화량을 나타냄")
        elif mean_corr > 0.3:
            print(f"   ⚠️  MIXED (상관계수: {mean_corr:.4f})")
            print(f"      - Delta와 absolute action이 혼재")
        else:
            print(f"   ❌ ABSOLUTE ACTION (상관계수: {mean_corr:.4f})")
            print(f"      - Action이 joint position 변화와 낮은 상관관계")
            print(f"      - 절대적인 목표 상태를 나타냄")
    
    # 각 파일별 상세 정보
    print(f"\n📁 파일별 상세 분석:")
    for i, result in enumerate(results[:5]):  # 처음 5개 파일만
        print(f"\n   {i+1}. {Path(result['file_path']).name}")
        print(f"      프레임 수: {result['n_frames']}")
        if result['correlations']:
            mean_file_corr = np.mean([c for c in result['correlations'] if not np.isnan(c)])
            print(f"      평균 상관계수: {mean_file_corr:.4f}")
    
    if len(results) > 5:
        print(f"   ... 및 {len(results)-5}개 파일 더")

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
    max_files = min(50, len(hdf5_files))
    print(f"📊 처음 {max_files}개 파일을 분석합니다...")
    
    # 각 파일 분석
    results = []
    for i, hdf5_file in enumerate(hdf5_files[:max_files]):
        try:
            result = analyze_action_patterns(hdf5_file)
            results.append(result)
            print(f"   ✅ {i+1}/{max_files}: {hdf5_file.name}")
        except Exception as e:
            print(f"   ❌ {i+1}/{max_files}: {hdf5_file.name} - 오류: {e}")
    
    if not results:
        print("❌ 분석할 수 있는 파일이 없습니다.")
        return
    
    # 결과 출력
    print_summary_analysis(results)
    
    # 그래프 생성
    try:
        plot_action_analysis(results)
    except Exception as e:
        print(f"⚠️  그래프 생성 중 오류: {e}")
        print("📊 그래프 없이 분석을 완료했습니다.")

if __name__ == "__main__":
    main() 