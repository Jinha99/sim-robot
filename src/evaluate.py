"""
Worm Robot Simulation - 학습된 모델 평가 (MAPPO)
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.mappo_agent import MAPPOAgent
from system import WormRobotSystem
from config import STATUS_WIN, STATUS_FAIL


def evaluate_model(model_path, num_episodes=20, num_robots=4, obstacles=None, 
                   moving_obstacles=None, verbose=True, termination_time=200):
    """
    학습된 MAPPO 모델 평가
    
    Args:
        model_path: 모델 파일 경로
        num_episodes: 평가 에피소드 수
        num_robots: 로봇 수
        obstacles: 장애물 리스트
        moving_obstacles: 움직이는 장애물 리스트
        verbose: 상세 출력 여부
        termination_time: 최대 스텝 수
    """
    print("=" * 70)
    print("📊 학습된 MAPPO 모델 평가")
    print("=" * 70)
    print(f"모델: {model_path}")
    print(f"평가 에피소드: {num_episodes}개")
    print(f"로봇 수: {num_robots}개")
    if obstacles:
        print(f"정적 장애물: {obstacles}")
    if moving_obstacles:
        print(f"움직이는 장애물: {len(moving_obstacles)}개")
    print("=" * 70)
    
    # MAPPO 에이전트 생성 및 로드
    agent = MAPPOAgent(
        state_dim=13,
        action_dim=4,
        num_agents=num_robots,
        learning_rate=3e-4,
        gamma=0.99,
        device="cpu"
    )
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    agent.load(model_path)
    
    # 평가 루프
    wins = 0
    fails = 0
    total_rewards = []
    total_steps = []
    
    print("\n평가 진행 중...\n")
    
    for episode in range(num_episodes):
        # 시스템 생성
        system = WormRobotSystem(
            rl_agent=None,
            num_robots=num_robots,
            obstacles=obstacles,
            moving_obstacles=moving_obstacles
        )
        
        episode_reward = 0.0
        step_count = 0
        
        while not system.is_done() and step_count < termination_time:
            # 현재 상태
            current_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    current_states[rid] = state
            
            # 행동 선택 (평가 모드 - deterministic)
            actions = {}
            for rid in current_states.keys():
                action = agent.get_action(current_states[rid], training=False)
                actions[rid] = action
            
            # 스텝 실행
            observations, rewards, done, status = system.step(actions)
            
            # 보상 합산
            for rid in rewards.keys():
                episode_reward += rewards[rid]
            
            step_count += 1
            
            if done:
                break
        
        # 통계 기록
        final_status = system.get_status()
        if final_status == STATUS_WIN:
            wins += 1
        elif final_status == STATUS_FAIL:
            fails += 1
        
        avg_reward = episode_reward / num_robots if num_robots > 0 else 0.0
        total_rewards.append(avg_reward)
        total_steps.append(step_count)
        
        if verbose:
            status_icon = "✅" if final_status == STATUS_WIN else "❌"
            print(f"{status_icon} Ep {episode+1:3d}: Reward={avg_reward:7.1f}, Steps={step_count:3d}, Status={final_status}")
    
    # 결과 출력
    win_rate = wins / num_episodes if num_episodes > 0 else 0.0
    avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0.0
    avg_steps = sum(total_steps) / len(total_steps) if total_steps else 0.0
    
    print("\n" + "=" * 70)
    print("📈 평가 결과")
    print("=" * 70)
    print(f"총 에피소드:     {num_episodes}개")
    print(f"성공:           {wins}회")
    print(f"실패:           {fails}회")
    print(f"승률:           {win_rate*100:.1f}%")
    print(f"평균 보상:       {avg_reward:.2f}")
    print(f"평균 스텝:       {avg_steps:.1f}")
    print("=" * 70)
    
    if win_rate > 0.5:
        print("\n✅ 모델이 성공적으로 학습되었습니다!")
    elif win_rate > 0:
        print("\n⚠️ 학습이 진행 중입니다. 더 많은 에피소드가 필요할 수 있습니다.")
    else:
        print("\n❌ 학습이 충분하지 않습니다. 더 많은 에피소드가 필요합니다.")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="학습된 MAPPO 모델 평가")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/mappo_phase2_4robots.pth",
        help="평가할 모델 파일 경로"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="평가 에피소드 수"
    )
    parser.add_argument(
        "--num-robots",
        type=int,
        default=4,
        help="로봇 수"
    )
    parser.add_argument(
        "--obstacles",
        type=str,
        default=None,
        help="정적 장애물 위치 (예: '(0,1),(-1,-1)')"
    )
    parser.add_argument(
        "--termination-time",
        type=int,
        default=200,
        help="최대 스텝 수"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    # 장애물 파싱
    obstacles = None
    if args.obstacles:
        try:
            obstacles = eval(f"[{args.obstacles}]")
        except:
            print(f"⚠️ 장애물 파싱 실패: {args.obstacles}")
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        num_robots=args.num_robots,
        obstacles=obstacles,
        moving_obstacles=None,
        verbose=args.verbose,
        termination_time=args.termination_time
    )


if __name__ == "__main__":
    main()

