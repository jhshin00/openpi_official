import h5py
import numpy as np
from pathlib import Path
import cv2

def check_mp4_fps(mp4_path):
    """MP4 파일에서 FPS 확인"""
    try:
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            print(f"❌ MP4 파일을 열 수 없습니다: {mp4_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        print(f"📹 MP4 파일: {mp4_path.name}")
        print(f"   FPS: {fps:.2f} Hz")
        print(f"   프레임 수: {frame_count}")
        print(f"   지속 시간: {duration:.2f}초 ({duration/60:.2f}분)")
        
        return fps, frame_count, duration
        
    except Exception as e:
        print(f"❌ MP4 파일 확인 중 오류: {e}")
        return None

def check_fps_from_hdf5(file_path):
    """HDF5 파일에서 FPS를 추정하는 함수"""
    
    with h5py.File(file_path, 'r') as f:
        print(f"HDF5 파일: {file_path}")
        print("=" * 50)
        
        # 데이터 길이 확인
        data_group = f['data']
        
        # 각 데이터셋의 길이 확인
        lengths = {}
        for key in data_group.keys():
            dataset = data_group[key]
            if len(dataset.shape) > 0:
                lengths[key] = dataset.shape[0]
                print(f"📄 {key}: {dataset.shape[0]} frames")
        
        # 모든 데이터셋의 길이가 같은지 확인
        unique_lengths = set(lengths.values())
        if len(unique_lengths) == 1:
            total_frames = list(unique_lengths)[0]
            print(f"\n✅ 모든 데이터셋이 {total_frames} 프레임으로 일치합니다.")
        else:
            print(f"\n⚠️  데이터셋 길이가 다릅니다: {lengths}")
            return
        
        # FPS 추정 방법들
        print(f"\n📊 FPS 추정:")
        
        # 1. 일반적인 로봇 데이터 FPS (10-30 Hz)
        common_fps = [10, 15, 20, 25, 30]
        print("일반적인 로봇 데이터 FPS 기준:")
        for fps in common_fps:
            duration = total_frames / fps
            print(f"  {fps} Hz: {duration:.2f}초 ({duration/60:.2f}분)")
        
        # 2. 파일명에서 시간 정보 추출 시도
        file_path = Path(file_path)
        parent_dir = file_path.parent.name
        print(f"\n📁 디렉토리명: {parent_dir}")
        
        # 시간 정보가 있는지 확인 (예: 0729_165009)
        if '_' in parent_dir and len(parent_dir) >= 6:
            try:
                time_part = parent_dir.split('_')[-1]
                if len(time_part) == 6:  # HHMMSS 형식
                    hour = int(time_part[:2])
                    minute = int(time_part[2:4])
                    second = int(time_part[4:6])
                    print(f"시간 정보: {hour:02d}:{minute:02d}:{second:02d}")
            except:
                pass
        
        # 3. 실제 녹화 시간 추정 (파일 크기 기반)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"\n📁 파일 크기: {file_size_mb:.1f} MB")
        
        # 이미지 데이터 크기 계산
        if 'base_rgb' in lengths:
            base_rgb_size = total_frames * 224 * 224 * 3  # uint8
            wrist_rgb_size = total_frames * 224 * 224 * 3  # uint8
            image_data_mb = (base_rgb_size + wrist_rgb_size) / (1024 * 1024)
            print(f"이미지 데이터 크기: {image_data_mb:.1f} MB")
            
            # 나머지 데이터 크기
            other_data_mb = file_size_mb - image_data_mb
            print(f"기타 데이터 크기: {other_data_mb:.1f} MB")
        
        print(f"\n💡 권장 FPS: 10 Hz (로봇 데이터의 일반적인 샘플링 레이트)")
        print(f"   - {total_frames} 프레임 / 10 Hz = {total_frames/10:.1f}초")
        
        return total_frames

def check_multiple_files():
    """여러 파일의 FPS를 비교"""
    
    base_dir = Path("/ssd2/openpi/datasets/ur3_datasets")
    hdf5_files = list(base_dir.glob("**/data.hdf5"))
    
    print(f"총 {len(hdf5_files)}개의 data.hdf5 파일을 찾았습니다.")
    print("=" * 60)
    
    frame_counts = []
    
    for i, file_path in enumerate(hdf5_files[:5]):  # 처음 5개만 확인
        print(f"\n{i+1}. {file_path.parent.name}")
        
        with h5py.File(file_path, 'r') as f:
            data_group = f['data']
            if 'joint_positions' in data_group:
                frame_count = data_group['joint_positions'].shape[0]
                frame_counts.append(frame_count)
                print(f"   프레임 수: {frame_count}")
    
    if frame_counts:
        print(f"\n📊 통계:")
        print(f"   평균 프레임 수: {np.mean(frame_counts):.1f}")
        print(f"   최소 프레임 수: {min(frame_counts)}")
        print(f"   최대 프레임 수: {max(frame_counts)}")
        print(f"   표준편차: {np.std(frame_counts):.1f}")

def compare_hdf5_and_mp4_fps(hdf5_path):
    """HDF5와 MP4 파일의 FPS 비교"""
    
    hdf5_path = Path(hdf5_path)
    parent_dir = hdf5_path.parent
    
    print(f"🔍 HDF5와 MP4 파일 FPS 비교")
    print("=" * 60)
    
    # HDF5 파일 정보
    hdf5_frames = check_fps_from_hdf5(hdf5_path)
    
    print("\n" + "=" * 60)
    
    # MP4 파일들 확인
    mp4_files = list(parent_dir.glob("*.mp4"))
    
    if not mp4_files:
        print("❌ MP4 파일을 찾을 수 없습니다.")
        return
    
    print(f"📹 발견된 MP4 파일들:")
    for mp4_file in mp4_files:
        print(f"   - {mp4_file.name}")
    
    print("\n" + "=" * 60)
    
    # 각 MP4 파일의 FPS 확인
    mp4_results = []
    for mp4_file in mp4_files:
        result = check_mp4_fps(mp4_file)
        if result:
            mp4_results.append((mp4_file.name, *result))
    
    # HDF5와 MP4 비교
    if hdf5_frames and mp4_results:
        print("\n" + "=" * 60)
        print("📊 HDF5 vs MP4 비교:")
        
        for mp4_name, mp4_fps, mp4_frames, mp4_duration in mp4_results:
            print(f"\n📹 {mp4_name}:")
            print(f"   MP4 FPS: {mp4_fps:.2f} Hz")
            print(f"   MP4 프레임: {mp4_frames}")
            print(f"   MP4 지속시간: {mp4_duration:.2f}초")
            
            # HDF5 데이터를 MP4 FPS로 계산한 지속시간
            hdf5_duration_with_mp4_fps = hdf5_frames / mp4_fps
            print(f"   HDF5 지속시간 (MP4 FPS 기준): {hdf5_duration_with_mp4_fps:.2f}초")
            
            # 차이 계산
            duration_diff = abs(mp4_duration - hdf5_duration_with_mp4_fps)
            print(f"   지속시간 차이: {duration_diff:.2f}초")
            
            if duration_diff < 1.0:
                print(f"   ✅ 일치! HDF5 FPS = {mp4_fps:.2f} Hz")
            else:
                print(f"   ⚠️  불일치 - 다른 FPS일 수 있음")

if __name__ == "__main__":
    # HDF5와 MP4 파일 비교
    test_file = "/ssd2/openpi/datasets/ur3_datasets/TASK_1_pick_up_the_bread_and_place_it_on_the_plate/0729_165009/data.hdf5"
    compare_hdf5_and_mp4_fps(test_file)
    
    print("\n" + "="*60)
    
    # 여러 파일 비교
    check_multiple_files() 