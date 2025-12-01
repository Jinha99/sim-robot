"""
MAPPO 기반 Curriculum Learning
로봇 수를 점진적으로 증가시키고, 이후 장애물 추가

Phase 0: 로봇 2개, 장애물 없음 (협력 학습 기초)
Phase 1: 로봇 3개, 장애물 없음 (협력 심화)
Phase 2: 로봇 4개, 장애물 없음 (최종 협력)
Phase 3: 로봇 4개 + 정적 장애물 1개
Phase 4: 로봇 4개 + 정적 장애물 3개
Phase 5: 로봇 4개 + 움직이는 장애물 1개
Phase 6: 로봇 4개 + 움직이는 장애물 2개
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.mappo_agent import MAPPOAgent
from system import WormRobotSystem
from moving_obstacle import create_moving_obstacles
from config import STATUS_WIN, STATUS_FAIL, STATUS_RUNNING


class MAPPOCurriculumTrainer:
    """MAPPO 기반 Curriculum Learning 트레이너"""
    
    def __init__(self, agent, log_interval=50, rollout_steps=2048):
        """
        Args:
            agent: MAPPOAgent 인스턴스
            log_interval: 로그 출력 간격 (에피소드)
            rollout_steps: 학습 전 수집할 경험 스텝 수
        """
        self.agent = agent
        self.log_interval = log_interval
        self.rollout_steps = rollout_steps
    
    def train_phase(self, phase_name, num_robots, obstacles=None, moving_obstacles=None,
                    num_episodes=5000, termination_time=80, success_threshold=0.3,
                    model_path=None):
        """
        단일 Phase 학습
        
        Args:
            phase_name: Phase 이름
            num_robots: 로봇 수
            obstacles: 정적 장애물 리스트 (None or [(x, y), ...])
            moving_obstacles: 움직이는 장애물 리스트
            num_episodes: 학습 에피소드 수
            termination_time: 최대 스텝
            success_threshold: 목표 성공률
            model_path: 저장할 모델 경로
        """
        print("\n" + "=" * 70)
        print(f"🎓 {phase_name} 학습 시작")
        print("=" * 70)
        print(f"로봇 수: {num_robots}개")
        print(f"정적 장애물: {obstacles if obstacles else '없음'}")
        print(f"움직이는 장애물: {len(moving_obstacles) if moving_obstacles else 0}개")
        print(f"에피소드: {num_episodes}")
        print(f"목표 성공률: {success_threshold * 100:.0f}%")
        print("=" * 70)
        
        # 시스템 생성 함수
        def create_system_fn():
            return WormRobotSystem(
                rl_agent=None,  # MAPPO는 외부에서 행동 선택
                num_robots=num_robots,
                obstacles=obstacles,
                moving_obstacles=moving_obstacles
            )
        
        # 중단된 학습 재개 확인
        resumed = False
        if model_path:
            tmp_interrupted = model_path.replace('.pth', '_tmp_interrupted.pth')
            tmp_error = model_path.replace('.pth', '_tmp_error.pth')
            
            if os.path.exists(tmp_interrupted):
                print(f"\n🔄 중단된 학습 발견! 이어서 진행합니다...")
                print(f"   모델 로드: {tmp_interrupted}")
                self.agent.load(tmp_interrupted)
                resumed = True
            elif os.path.exists(tmp_error):
                print(f"\n🔄 오류로 중단된 학습 발견! 이어서 진행합니다...")
                print(f"   모델 로드: {tmp_error}")
                self.agent.load(tmp_error)
                resumed = True
        
        if resumed:
            print(f"   ✅ 이전 학습 상태에서 재개합니다!")
        
        # 통계
        stats = {
            "episode_rewards": [],
            "episode_steps": [],
            "episode_losses": [],
            "success_count": 0,
            "fail_count": 0
        }
        
        print(f"\n🚀 학습 시작!\n")
        
        best_success_rate = 0.0
        total_steps = 0
        
        # 학습 루프
        try:
            for episode in range(num_episodes):
                episode_reward, episode_steps, status = self._run_episode(
                    create_system_fn, termination_time
                )
                
                total_steps += episode_steps
                
                # 통계 기록
                stats["episode_rewards"].append(episode_reward)
                stats["episode_steps"].append(episode_steps)
                
                if status == STATUS_WIN:
                    stats["success_count"] += 1
                elif status == STATUS_FAIL:
                    stats["fail_count"] += 1
                
                # 일정 스텝마다 학습
                if total_steps >= self.rollout_steps:
                    actor_loss, critic_loss, entropy = self.agent.train()
                    stats["episode_losses"].append(actor_loss)
                    total_steps = 0
                
                # 로그 출력
                if (episode + 1) % self.log_interval == 0:
                    recent = min(self.log_interval, len(stats["episode_rewards"]))
                    avg_reward = sum(stats["episode_rewards"][-recent:]) / recent
                    avg_steps = sum(stats["episode_steps"][-recent:]) / recent
                    avg_loss = sum(stats["episode_losses"][-recent:]) / recent if stats["episode_losses"] else 0
                    
                    # 최근 성공률 계산
                    recent_episodes = stats["success_count"] + stats["fail_count"]
                    recent_success_rate = stats["success_count"] / recent_episodes if recent_episodes > 0 else 0
                    
                    print(
                        f"Ep {episode + 1:5d}/{num_episodes} | "
                        f"Reward: {avg_reward:7.1f} | "
                        f"Steps: {avg_steps:4.1f} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Success: {recent_success_rate*100:4.1f}% | "
                        f"Win: {stats['success_count']:4d}"
                    )
                    
                    # 최고 성공률 갱신
                    if recent_success_rate > best_success_rate:
                        best_success_rate = recent_success_rate
                        if model_path:
                            self.agent.save(model_path)
                            print(f"   ✅ 새 최고 성공률! 모델 저장: {model_path}")
                
                # 조기 종료 조건 체크
                if episode > 1000 and episode % 500 == 0:
                    total_episodes = stats["success_count"] + stats["fail_count"]
                    current_success_rate = stats["success_count"] / total_episodes if total_episodes > 0 else 0
                    
                    if current_success_rate >= success_threshold:
                        print(f"\n🎉 목표 달성! (전체 성공률: {current_success_rate*100:.1f}%)")
                        break
        
        except KeyboardInterrupt:
            print(f"\n\n⚠️ 사용자가 학습을 중단했습니다!")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_interrupted.pth')
                self.agent.save(tmp_path)
                print(f"   💾 임시 모델 저장: {tmp_path}")
                print(f"   현재까지 진행: {episode + 1}/{num_episodes} 에피소드")
            raise
        
        except Exception as e:
            print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_error.pth')
                self.agent.save(tmp_path)
                print(f"   💾 임시 모델 저장: {tmp_path}")
            raise
        
        # 최종 저장
        if model_path:
            self.agent.save(model_path)
        
        # 최종 통계
        total_episodes = stats["success_count"] + stats["fail_count"]
        final_success_rate = stats["success_count"] / total_episodes if total_episodes > 0 else 0
        
        print("\n" + "=" * 70)
        print(f"✅ {phase_name} 완료!")
        print("=" * 70)
        print(f"총 성공: {stats['success_count']}")
        print(f"총 실패: {stats['fail_count']}")
        print(f"최종 성공률: {final_success_rate * 100:.1f}%")
        print(f"최고 성공률: {best_success_rate * 100:.1f}%")
        print("=" * 70)
        
        return stats, final_success_rate
    
    def _run_episode(self, create_system_fn, termination_time):
        """에피소드 실행 (MAPPO용)"""
        system = create_system_fn()
        num_robots = len(system.robots)

        # 목표 위치 설정 (승리 조건과 동일하게)
        if num_robots <= 2:
            target_positions = {(0, 1), (0, -1)}
        else:
            target_positions = {(0, 1), (1, 0), (0, -1), (-1, 0)}

        # 각 로봇의 이전 거리 저장을 위한 딕셔너리
        prev_distances = {}
        for rid in range(num_robots):
            # 각 로봇의 앞발에서 가장 가까운 목표 지점까지의 초기 거리 계산
            head_pos = system.environment.state.robot_positions[rid]["head"]

            min_dist = min([abs(head_pos[0] - tx) + abs(head_pos[1] - ty) for tx, ty in target_positions])
            prev_distances[rid] = min_dist

        episode_reward = 0.0
        step_count = 0

        while not system.is_done() and step_count < termination_time:
            # 현재 상태
            current_states = {}
            for rid in range(num_robots):
                if rid in system.environment.state.robot_positions:
                    state = system.get_state_for_robot(rid)
                    current_states[rid] = state

            # 행동 선택 (MAPPO)
            actions = {}
            log_probs = {}
            values = {}

            for rid in current_states.keys():
                action, log_prob, value = self.agent.get_action(current_states[rid], training=True)
                actions[rid] = action
                log_probs[rid] = log_prob
                values[rid] = value

            # 스텝 실행
            observations, _, done, status = system.step(actions) # 기존 rewards 변수는 사용 안함
            # 🔹 현재 각 로봇의 head 위치와, 타겟 셀 점령 개수 계산
            heads = {
                rid: system.environment.state.robot_positions[rid]["head"]
                for rid in current_states.keys()
            }

            # target_positions는 함수 맨 위에서 승리조건과 동일하게 정의한 그 집합
            #   num_robots <= 2 → {(0,1), (0,-1)}
            #   num_robots  > 2 → {(0,1), (1,0), (0,-1), (-1,0)}
            occupied_targets = set()
            for h in heads.values():
                if h in target_positions:
                    occupied_targets.add(h)
            num_occupied = len(occupied_targets)  # 현재 타겟 칸 몇 개가 채워졌는지 (0~4)

            # 보상 재설계 (Reward Shaping)
            step_total_reward = 0.0
            for rid in current_states.keys():
                robot_reward = 0.0

                # 1. 스텝마다 작은 페널티 (최소 스텝 유도)
                robot_reward -= 1.0

                # 2. 목표에 가까워지는 것에 대한 보상
                head_pos = system.environment.state.robot_positions[rid]["head"]
                if head_pos in target_positions:
                    robot_reward += 3.0

                min_dist_to_target = min([abs(head_pos[0] - tx) + abs(head_pos[1] - ty) for tx, ty in target_positions])

                distance_diff = prev_distances[rid] - min_dist_to_target
                if distance_diff > 0:
                    robot_reward += 10.0  # 목표에 가까워졌을 때 큰 보상
                elif distance_diff < 0:
                    robot_reward -= 5.0   # 목표에서 멀어졌을 때 페널티

                prev_distances[rid] = min_dist_to_target

                # 2-1. 타겟 칸 점령 개수에 따른 추가 보상 (특히 4로봇에서 효과적)
                #   - 타겟 칸 1개 채워져 있으면 +2
                #   - 4개 다 채워져 있으면 +8
                #   → step penalty(-1)를 어느 정도 상쇄해 주면서도 너무 크지 않게

                # 타겟 칸 점령 개수 보상
                num_robots = len(current_states)
                if num_robots == 4:
                    # 4로봇일 때는 타겟 점령 영향 더 크게
                    robot_reward += num_occupied * 4.0  # 0~16
                else:
                    robot_reward += num_occupied * 2.0  # 기존 값 유지


                # 3. 최종 성공/실패에 대한 큰 보상/페널티
                if done:
                    if status == STATUS_WIN:
                        robot_reward += 500.0
                    elif status == STATUS_FAIL:
                        robot_reward -= 500.0

                # 경험 저장
                self.agent.store_transition(
                    current_states[rid],
                    actions[rid],
                    robot_reward,
                    values[rid],
                    log_probs[rid],
                    float(done)
                )
                step_total_reward += robot_reward

            episode_reward += step_total_reward
            step_count += 1

            if done:
                break

        avg_reward = episode_reward / num_robots if num_robots > 0 else 0.0
        final_status = system.get_status()

        return avg_reward, step_count, final_status


def main():
    print("\n" + "=" * 70)
    print("🤖 MAPPO 기반 Curriculum Learning")
    print("=" * 70)
    print("전략:")
    print("  Phase 0: 로봇 2개, 장애물 없음 (협력 학습 기초)")
    print("  Phase 1: 로봇 3개, 장애물 없음 (협력 심화)")
    print("  Phase 2: 로봇 4개, 장애물 없음 (최종 협력)")
    print("  Phase 3: 로봇 4개 + 정적 장애물 1개")
    print("  Phase 4: 로봇 4개 + 정적 장애물 3개")
    print("  Phase 5: 로봇 4개 + 움직이는 장애물 1개")
    print("  Phase 6: 로봇 4개 + 움직이는 장애물 2개")
    print("=" * 70)

    # MAPPO 에이전트 생성 (한 번만!)
    agent = MAPPOAgent(
        state_dim=13,
        action_dim=4,  # 전진, 시계, 반시계, STAY
        num_agents=4,  # 최대 로봇 수
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device="cpu"
    )
    agent.load("outputs/mappo_phase2_4robots.pth")
    print("🔥 Phase 0 모델 로드 완료! 이제 Phase 1부터 시작합니다.")

    # 트레이너 생성
    trainer = MAPPOCurriculumTrainer(
        agent=agent,
        log_interval=50,
        rollout_steps=2048
    )
    '''
    # Phase 0: 로봇 2개, 장애물 없음 (협력 학습 기초)
    try:
        phase0_stats, phase0_success = trainer.train_phase(
            phase_name="Phase 0: 로봇 2개 협력 기초",
            num_robots=2,
            obstacles=None,
            num_episodes=10000,
            termination_time=100,
            success_threshold=0.4,
            model_path="outputs/mappo_phase0_2robots.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    if phase0_success < 0.15:
        print("\n❌ Phase 0 실패! 협력 학습이 제대로 되지 않았습니다.")
        print("   하이퍼파라미터를 재조정하거나 에피소드 수를 늘려야 합니다.")
        return
    
    # Phase 1: 로봇 3개, 장애물 없음 (협력 심화)
    try:
        phase1_stats, phase1_success = trainer.train_phase(
            phase_name="Phase 1: 로봇 3개 협력 심화",
            num_robots=3,
            obstacles=None,
            num_episodes=15000,
            termination_time=120,
            success_threshold=0.3,
            model_path="outputs/mappo_phase1_3robots.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    if phase1_success < 0.1:
        print("\n⚠️ Phase 1 성공률 낮음. 그래도 Phase 2로 진행합니다.")
    '''
    # Phase 2: 로봇 4개, 장애물 없음 (최종 협력)
    try:
        phase2_stats, phase2_success = trainer.train_phase(
            phase_name="Phase 2: 로봇 4개 최종 협력",
            num_robots=4,
            obstacles=None,
            num_episodes=25000,
            termination_time=150,
            success_threshold=0.25,
            model_path="outputs/mappo_phase2_4robots.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    if phase2_success < 0.08:
        print("\n⚠️ Phase 2 성공률 낮음. 그래도 Phase 3으로 진행합니다.")
    '''
    # Phase 3: 로봇 4개 + 정적 장애물 1개
    try:
        phase3_stats, phase3_success = trainer.train_phase(
            phase_name="Phase 3: 로봇 4개 + 정적 장애물 1개",
            num_robots=4,
            obstacles=[(2, 2)],  # 모서리 1개
            num_episodes=20000,
            termination_time=150,
            success_threshold=0.2,
            model_path="outputs/mappo_phase3_4robots_obs1.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    # Phase 4: 로봇 4개 + 정적 장애물 3개
    try:
        phase4_stats, phase4_success = trainer.train_phase(
            phase_name="Phase 4: 로봇 4개 + 정적 장애물 3개",
            num_robots=4,
            obstacles=[(0, 1), (-1, -1), (1, 0)],
            num_episodes=30000,
            termination_time=150,
            success_threshold=0.15,
            model_path="outputs/mappo_phase4_4robots_obs3.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    # Phase 5: 로봇 4개 + 움직이는 장애물 1개
    moving_obs_1 = create_moving_obstacles(count=1)
    try:
        phase5_stats, phase5_success = trainer.train_phase(
            phase_name="Phase 5: 로봇 4개 + 움직이는 장애물 1개",
            num_robots=4,
            obstacles=None,
            moving_obstacles=moving_obs_1,
            num_episodes=25000,
            termination_time=180,
            success_threshold=0.12,
            model_path="outputs/mappo_phase5_4robots_moving1.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return

    # Phase 6: 로봇 4개 + 움직이는 장애물 2개
    moving_obs_2 = create_moving_obstacles(count=2)
    try:
        phase6_stats, phase6_success = trainer.train_phase(
            phase_name="Phase 6: 로봇 4개 + 움직이는 장애물 2개",
            num_robots=4,
            obstacles=None,
            moving_obstacles=moving_obs_2,
            num_episodes=35000,
            termination_time=200,
            success_threshold=0.1,
            model_path="outputs/mappo_phase6_4robots_moving2.pth"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 학습이 중단되었습니다. 프로그램을 종료합니다.")
        return
    '''
    # 최종 요약
    print("\n" + "=" * 70)
    print("🎉 MAPPO Curriculum Learning 완료!")
    print("=" * 70)
    #print(f"Phase 0 (2개, 장애물 없음):           {phase0_success*100:5.1f}%")
    print(f"Phase 1 (3개, 장애물 없음):           {phase1_success*100:5.1f}%")
    print(f"Phase 2 (4개, 장애물 없음):           {phase2_success*100:5.1f}%")
    print(f"Phase 3 (4개, 정적 1개):              {phase3_success*100:5.1f}%")
    print(f"Phase 4 (4개, 정적 3개):              {phase4_success*100:5.1f}%")
    print(f"Phase 5 (4개, 움직임 1개):            {phase5_success*100:5.1f}%")
    print(f"Phase 6 (4개, 움직임 2개):            {phase6_success*100:5.1f}%")
    print("=" * 70)
    print("\n저장된 모델:")
    print("  outputs/mappo_phase0_2robots.pth")
    print("  outputs/mappo_phase1_3robots.pth")
    print("  outputs/mappo_phase2_4robots.pth")
    print("  outputs/mappo_phase3_4robots_obs1.pth")
    print("  outputs/mappo_phase4_4robots_obs3.pth")
    print("  outputs/mappo_phase5_4robots_moving1.pth")
    print("  outputs/mappo_phase6_4robots_moving2.pth")
    print("\n평가 명령어:")
    print("  python3.11 evaluate.py --model outputs/mappo_phase2_4robots.pth --num-robots 4")
    print("  python3.11 evaluate.py --model outputs/mappo_phase4_4robots_obs3.pth --num-robots 4 --obstacles '(0,1),(-1,-1),(1,0)'")
    print("=" * 70)


if __name__ == "__main__":
    main()
