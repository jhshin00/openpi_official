#!/usr/bin/env python3
"""
실제 데이터에서 Task 이름 변환 확인
"""

from pathlib import Path

def clean_task_name(task_name):
    """Task 이름을 깔끔하게 변환"""
    if task_name.startswith("TASK"):
        # TASK 부분 완전히 제거 (TASK, TASK_4, TASK1_ 등)
        if "_" in task_name:
            # 첫 번째 언더스코어 이후 부분만 가져오기
            task_name = task_name.split("_", 1)[1]
            # 만약 숫자로 시작한다면 그 숫자도 제거
            if task_name and task_name[0].isdigit():
                # 숫자 부분을 건너뛰고 다음 언더스코어 이후부터
                if "_" in task_name:
                    task_name = task_name.split("_", 1)[1]
                else:
                    task_name = ""  # 숫자만 있다면 빈 문자열
        # 나머지 언더스코어를 공백으로 변환
        task_name = task_name.replace("_", " ")
    return task_name

def test_real_tasks():
    """실제 데이터에서 task 이름 확인"""
    
    base_dir = Path("/ssd2/openpi/datasets/ur3_datasets")
    hdf5_files = list(base_dir.glob("**/data.hdf5"))
    
    print(f"📁 실제 데이터에서 Task 이름 변환 확인:")
    print("=" * 60)
    
    tasks = set()
    
    for i, file_path in enumerate(hdf5_files[:10]):  # 처음 10개만 확인
        path_parts = file_path.parts
        task = "ur3_task"  # 기본값
        
        for part in path_parts:
            if part.startswith("TASK"):
                task = part
                break
        
        cleaned_task = clean_task_name(task)
        tasks.add((task, cleaned_task))
        print(f"{i+1:2d}. {file_path.parent.name}")
        print(f"    원본: {task}")
        print(f"    변환: {cleaned_task}")
        print()
    
    print(f"📊 발견된 Task 종류:")
    for original, cleaned in sorted(tasks):
        print(f"  - {original} → {cleaned}")

if __name__ == "__main__":
    test_real_tasks() 