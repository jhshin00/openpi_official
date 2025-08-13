import h5py
from pathlib import Path
import os

def calculate_fps_from_timestamps(file_path):
    """시간 정보를 이용해서 FPS 계산"""
    
    file_path = Path(file_path)
    parent_dir = file_path.parent.name
    
    print(f"📁 디렉토리: {parent_dir}")
    
    # HDF5 파일에서 프레임 수 확인
    with h5py.File(file_path, 'r') as f:
        frame_count = f['data']['joint_positions'].shape[0]
        print(f"📊 프레임 수: {frame_count}")
    
    # 시간 정보 파싱 (예: 0729_165009)
    if '_' in parent_dir and len(parent_dir) >= 6:
        try:
            time_part = parent_dir.split('_')[-1]
            if len(time_part) == 6:  # HHMMSS 형식
                hour = int(time_part[:2])
                minute = int(time_part[2:4])
                second = int(time_part[4:6])
                
                print(f"⏰ 시간 정보: {hour:02d}:{minute:02d}:{second:02d}")
                
                # 실제 녹화 시간을 알려면 다음 중 하나가 필요합니다:
                # 1. 시작 시간과 종료 시간
                # 2. 녹화 지속 시간
                # 3. 다음 파일과의 시간 차이
                
                print(f"\n💡 FPS 계산을 위해서는 다음 정보가 필요합니다:")
                print(f"   1. 녹화 시작 시간")
                print(f"   2. 녹화 종료 시간")
                print(f"   3. 또는 녹화 지속 시간")
                
                return None
                
        except:
            pass
    
    print(f"❌ 시간 정보를 파싱할 수 없습니다.")
    return None

def estimate_fps_from_common_values(frame_count):
    """일반적인 FPS 값들로 추정"""
    
    print(f"\n📊 {frame_count} 프레임에 대한 FPS 추정:")
    
    common_fps = [10, 15, 20, 25, 30, 60]
    
    for fps in common_fps:
        duration = frame_count / fps
        print(f"  {fps} Hz: {duration:.2f}초 ({duration/60:.2f}분)")
    
    print(f"\n💡 로봇 데이터의 일반적인 FPS:")
    print(f"   - 로봇 제어: 10-30 Hz")
    print(f"   - 카메라: 10-30 Hz")
    print(f"   - 센서 데이터: 10-100 Hz")

if __name__ == "__main__":
    test_file = "/ssd2/openpi/datasets/ur3_datasets/TASK_1_pick_up_the_bread_and_place_it_on_the_plate/0729_165009/data.hdf5"
    
    # 시간 정보로 FPS 계산 시도
    fps = calculate_fps_from_timestamps(test_file)
    
    # 일반적인 값으로 추정
    with h5py.File(test_file, 'r') as f:
        frame_count = f['data']['joint_positions'].shape[0]
    
    estimate_fps_from_common_values(frame_count) 