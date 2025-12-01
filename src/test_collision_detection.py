"""
테스트: 동일 칸 충돌 감지 및 패널티 시스템
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from system import WormRobotSystem
from config import STATUS_RUNNING, STATUS_WIN, STATUS_FAIL


def test_collision_detection():
    """충돌 감지 테스트"""
    print("\n" + "=" * 70)
    print("🧪 동일 칸 충돌 감지 테스트")
    print("=" * 70)

    # 2개 로봇으로 간단한 테스트
    system = WormRobotSystem(
        rl_agent=None,
        num_robots=2,
        obstacles=None,
        moving_obstacles=None
    )

    print("\n초기 상태:")
    for rid in range(2):
        pos = system.environment.state.robot_positions[rid]
        print(f"  Robot {rid}: Head={pos['head']}, Tail={pos['tail']}, Dir={pos['direction']}")

    # 시나리오 1: 두 로봇이 모두 앞으로 이동
    print("\n" + "-" * 70)
    print("시나리오 1: 두 로봇 모두 MOVE 실행")
    print("-" * 70)

    # 의도적으로 충돌을 만들기 위해 여러 번 실행
    for step in range(5):
        print(f"\n[Step {step + 1}]")

        # 두 로봇 모두 MOVE (action_idx=0)
        actions = {0: 0, 1: 0}

        observations, rewards, done, status = system.step(actions)

        print(f"  Rewards: {rewards}")

        # 충돌 패널티가 있는지 확인
        if any(r < 0 for r in rewards.values()):
            print(f"  ⚠️ 충돌 감지! 패널티 부여됨")
            for rid, reward in rewards.items():
                if reward < 0:
                    print(f"    - Robot {rid}: {reward} 패널티")
        else:
            print(f"  ✅ 충돌 없음")

        # 현재 위치 출력
        for rid in range(2):
            if rid in system.environment.state.robot_positions:
                pos = system.environment.state.robot_positions[rid]
                print(f"  Robot {rid}: Head={pos['head']}, Tail={pos['tail']}")

        if done:
            print(f"\n게임 종료: {status}")
            break

    print("\n" + "=" * 70)
    print("✅ 테스트 완료!")
    print("=" * 70)


def test_collision_scenarios():
    """다양한 충돌 시나리오 테스트"""
    print("\n" + "=" * 70)
    print("🧪 다양한 충돌 시나리오 테스트")
    print("=" * 70)

    # 3개 로봇
    system = WormRobotSystem(
        rl_agent=None,
        num_robots=3,
        obstacles=None,
        moving_obstacles=None
    )

    print("\n초기 상태:")
    for rid in range(3):
        pos = system.environment.state.robot_positions[rid]
        print(f"  Robot {rid}: Head={pos['head']}, Tail={pos['tail']}")

    # 시나리오: 3개 로봇 중 2개만 충돌
    print("\n" + "-" * 70)
    print("시나리오: 로봇들이 다양한 행동 수행")
    print("-" * 70)

    for step in range(3):
        print(f"\n[Step {step + 1}]")

        # Robot 0: MOVE, Robot 1: ROTATE_CW, Robot 2: MOVE
        actions = {0: 0, 1: 1, 2: 0}

        observations, rewards, done, status = system.step(actions)

        print(f"  Actions: Robot 0=MOVE, Robot 1=ROTATE_CW, Robot 2=MOVE")
        print(f"  Rewards: {rewards}")

        collision_detected = any(r < 0 for r in rewards.values())
        print(f"  충돌 감지: {'예' if collision_detected else '아니오'}")

        for rid in range(3):
            if rid in system.environment.state.robot_positions:
                pos = system.environment.state.robot_positions[rid]
                print(f"  Robot {rid}: Head={pos['head']}, Tail={pos['tail']}")

        if done:
            print(f"\n게임 종료: {status}")
            break

    print("\n" + "=" * 70)
    print("✅ 시나리오 테스트 완료!")
    print("=" * 70)


if __name__ == "__main__":
    test_collision_detection()
    print("\n")
    test_collision_scenarios()
