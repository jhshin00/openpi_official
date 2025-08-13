#!/usr/bin/env python3
"""
Task 이름 변환 테스트
"""

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

def test_task_cleaning():
    """Task 이름 변환 테스트"""
    
    test_cases = [
        "TASK1_put_the_bread_on_the_plate",
        "TASK2_pick_up_the_gray_pot",
        "TASK3_move_the_red_cup",
        "TASK_4_pick_up_the_pink_cup_and_place_it_between_the_stove_and_the_gray_pot",
        "TASK_10_complex_manipulation_task",
        "ur3_task",  # 기본값
    ]
    
    print("🔍 Task 이름 변환 테스트:")
    print("=" * 50)
    
    for original in test_cases:
        cleaned = clean_task_name(original)
        print(f"📝 {original}")
        print(f"   → {cleaned}")
        print()

if __name__ == "__main__":
    test_task_cleaning() 