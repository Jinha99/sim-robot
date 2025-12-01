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
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl.mappo_agent import MAPPOAgent
from system import WormRobotSystem
from moving_obstacle import create_moving_obstacles
from config import STATUS_WIN, STATUS_FAIL, STATUS_RUNNING


class MAPPOCurriculumTrainer:
    """MAPPO 기반 Curriculum Learning 트레이너"""
    
    def __init__(self, agent, log_interval=50, rollout_steps=2048):
        self.agent = agent
        self.log_interval = log_interval
        self.rollout_steps = rollout_steps
    
    def _save_state(self, path, best_rate):
        state_path = path.replace('.pth', '_state.json')
        try:
            with open(state_path, 'w') as f:
                json.dump({'best_success_rate': best_rate}, f)
        except IOError as e:
            print(f"   ⚠️ 상태 파일 저장 실패: {e}")

    def _load_state(self, path):
        state_path = path.replace('.pth', '_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    best_rate = state.get('best_success_rate', 0.0)
                    print(f"   📊 이전 최고 성공률 로드: {best_rate*100:.1f}%")
                    return best_rate
            except (json.JSONDecodeError, IOError) as e:
                print(f"   ⚠️ 상태 파일 로드 실패: {e}. 최고 성공률을 0.0에서 시작합니다.")
        return 0.0

    def train_phase(self, phase_name, num_robots, obstacles=None, moving_obstacles=None,
                    num_episodes=5000, termination_time=80, success_threshold=0.3,
                    model_path=None):
        print("\n" + "=" * 70)
        print(f"🎓 {phase_name} 학습 시작")
        print("=" * 70)
        print(f"로봇 수: {num_robots}개, 최대 스텝: {termination_time}")
        print(f"정적 장애물: {obstacles if obstacles else '없음'}")
        print(f"움직이는 장애물: {len(moving_obstacles) if moving_obstacles else 0}개")
        print(f"에피소드: {num_episodes}")
        print("=" * 70)
        
        def create_system_fn():
            return WormRobotSystem(
                rl_agent=None, num_robots=num_robots, obstacles=obstacles,
                moving_obstacles=moving_obstacles
            )
        
        best_success_rate = 0.0
        if model_path:
            load_path = None
            tmp_interrupted = model_path.replace('.pth', '_tmp_interrupted.pth')
            tmp_error = model_path.replace('.pth', '_tmp_error.pth')
            
            if os.path.exists(tmp_interrupted):
                load_path = tmp_interrupted
            elif os.path.exists(tmp_error):
                load_path = tmp_error
            
            if load_path:
                print(f"\n🔄 중단/오류된 학습 발견! 이어서 진행합니다...")
                self.agent.load(load_path)
                best_success_rate = self._load_state(load_path)
        
        stats = {
            "episode_rewards": [], "episode_steps": [], "episode_losses": [],
            "success_count": 0, "fail_count": 0
        }
        
        print(f"\n🚀 학습 시작!\n")
        total_steps = 0
        
        try:
            for episode in range(num_episodes):
                episode_reward, episode_steps, status = self._run_episode(
                    create_system_fn, termination_time
                )
                
                total_steps += episode_steps
                stats["episode_rewards"].append(episode_reward)
                stats["episode_steps"].append(episode_steps)
                
                if status == STATUS_WIN: stats["success_count"] += 1
                elif status == STATUS_FAIL: stats["fail_count"] += 1
                
                if total_steps >= self.rollout_steps:
                    actor_loss, _, _ = self.agent.train()
                    if actor_loss is not None: stats["episode_losses"].append(actor_loss)
                    total_steps = 0
                
                if (episode + 1) % self.log_interval == 0:
                    recent_count = stats["success_count"] + stats["fail_count"]
                    if recent_count > 0:
                        recent_success_rate = stats["success_count"] / recent_count
                        avg_reward = sum(stats["episode_rewards"]) / recent_count
                        avg_steps = sum(stats["episode_steps"]) / recent_count
                        avg_loss = sum(stats["episode_losses"]) / len(stats["episode_losses"]) if stats["episode_losses"] else 0
                        
                        print(
                            f"Ep {episode + 1:5d}/{num_episodes} | "
                            f"Reward: {avg_reward:7.1f} | "
                            f"Steps: {avg_steps:4.1f} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Success: {recent_success_rate*100:4.1f}% | "
                            f"Best: {best_success_rate*100:4.1f}% | "
                            f"Win: {stats['success_count']:4d}"
                        )
                        
                        if recent_success_rate > best_success_rate:
                            best_success_rate = recent_success_rate
                            if model_path:
                                self.agent.save(model_path)
                                self._save_state(model_path, best_success_rate)
                                print(f"   ✅ 새 최고 성공률 달성! 모델 저장: {model_path}")
                        
                        stats = {
                            "episode_rewards": [], "episode_steps": [], "episode_losses": [],
                            "success_count": 0, "fail_count": 0
                        }

        except KeyboardInterrupt:
            print(f"\n\n⚠️ 사용자가 학습을 중단했습니다!")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_interrupted.pth')
                self.agent.save(tmp_path)
                self._save_state(tmp_path, best_success_rate)
                print(f"   💾 임시 모델 저장: {tmp_path}")
            return
        
        except Exception as e:
            print(f"\n\n❌ 예상치 못한 오류 발생: {e}")
            if model_path:
                tmp_path = model_path.replace('.pth', '_tmp_error.pth')
                self.agent.save(tmp_path)
                self._save_state(tmp_path, best_success_rate)
                print(f"   💾 임시 모델 저장: {tmp_path}")
            raise
        
        if model_path:
            self.agent.save(model_path)
            self._save_state(model_path, best_success_rate)
        
        print("\n" + "=" * 70)
        print(f"✅ {phase_name} 완료! (최고 성공률: {best_success_rate * 100:.1f}%)")
        print("=" * 70)
        
        return best_success_rate

    def _run_episode(self, create_system_fn, termination_time):
        system = create_system_fn()
        num_robots = len(system.robots)

        if num_robots <= 2:
            target_positions = {(0, 1), (0, -1)}
        else:
            target_positions = {(0, 1), (1, 0), (0, -1), (-1, 0)}

        prev_distances = {}
        for rid in range(num_robots):
            head_pos = system.environment.state.robot_positions[rid]["head"]
            min_dist = min([abs(head_pos[0] - tx) + abs(head_pos[1] - ty) for tx, ty in target_positions])
            prev_distances[rid] = min_dist

        episode_reward, step_count = 0.0, 0
        
        while not system.is_done() and step_count < termination_time:
            current_states = {rid: system.get_state_for_robot(rid) for rid in range(num_robots) if rid in system.environment.state.robot_positions}
            
            actions, log_probs, values = {}, {}, {}
            for rid, state in current_states.items():
                action, log_prob, value = self.agent.get_action(state, training=True)
                actions[rid], log_probs[rid], values[rid] = action, log_prob, value
            
            _, _, done, status = system.step(actions)

            step_total_reward = 0.0
            for rid in current_states.keys():
                robot_reward = -10.0  # 스텝 페널티 대폭 강화
                
                head_pos = system.environment.state.robot_positions[rid]["head"]
                min_dist_to_target = min([abs(head_pos[0] - tx) + abs(head_pos[1] - ty) for tx, ty in target_positions])
                
                distance_diff = prev_distances[rid] - min_dist_to_target
                if distance_diff > 0: robot_reward += 10.0
                elif distance_diff < 0: robot_reward -= 5.0
                prev_distances[rid] = min_dist_to_target

                if actions[rid] == 3 and min_dist_to_target > 1:
                    robot_reward -= 3.0

                if done:
                    if status == STATUS_WIN:
                        robot_reward += 500.0
                        robot_reward += (termination_time - step_count) * 5.0  # 효율성 보너스
                    elif status == STATUS_FAIL:
                        robot_reward -= 500.0

                self.agent.store_transition(
                    current_states[rid], actions[rid], robot_reward,
                    values[rid], log_probs[rid], float(done)
                )
                step_total_reward += robot_reward
            
            episode_reward += step_total_reward
            step_count += 1
            if done: break
        
        return (episode_reward / num_robots if num_robots > 0 else 0.0), step_count, system.get_status()


def main():
    print("\n" + "=" * 70)
    print("🤖 MAPPO 기반 Curriculum Learning")
    print("=" * 70)
    
    agent = MAPPOAgent(
        state_dim=13, action_dim=4, num_agents=4, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.01,
        value_loss_coef=0.5, max_grad_norm=0.5, device="cpu"
    )
    
    trainer = MAPPOCurriculumTrainer(agent=agent, log_interval=50, rollout_steps=2048)
    
    phases = [
        {"name": "Phase 0: 로봇 2개 협력 기초", "robots": 2, "episodes": 10000, "time": 80, "model": "outputs/mappo_phase0_2robots.pth"},
        {"name": "Phase 1: 로봇 3개 협력 심화", "robots": 3, "episodes": 15000, "time": 80, "model": "outputs/mappo_phase1_3robots.pth"},
        {"name": "Phase 2: 로봇 4개 최종 협력", "robots": 4, "episodes": 25000, "time": 80, "model": "outputs/mappo_phase2_4robots.pth"},
        {"name": "Phase 3: 로봇 4개 + 정적 장애물 1개", "robots": 4, "episodes": 20000, "time": 90, "model": "outputs/mappo_phase3_4robots_obs1.pth", "obstacles": [(2, 2)]},
        {"name": "Phase 4: 로봇 4개 + 정적 장애물 3개", "robots": 4, "episodes": 30000, "time": 90, "model": "outputs/mappo_phase4_4robots_obs3.pth", "obstacles": [(0, 1), (-1, -1), (1, 0)]},
        {"name": "Phase 5: 로봇 4개 + 움직이는 장애물 1개", "robots": 4, "episodes": 25000, "time": 90, "model": "outputs/mappo_phase5_4robots_moving1.pth", "moving_obs_count": 1},
        {"name": "Phase 6: 로봇 4개 + 움직이는 장애물 2개", "robots": 4, "episodes": 35000, "time": 90, "model": "outputs/mappo_phase6_4robots_moving2.pth", "moving_obs_count": 2}
    ]
    
    final_success_rates = {}

    for i, p in enumerate(phases):
        moving_obstacles = create_moving_obstacles(count=p.get("moving_obs_count", 0))
        
        try:
            success_rate = trainer.train_phase(
                phase_name=p["name"], num_robots=p["robots"], obstacles=p.get("obstacles"),
                moving_obstacles=moving_obstacles, num_episodes=p["episodes"],
                termination_time=p["time"], model_path=p["model"]
            )
            final_success_rates[p["name"]] = success_rate

            if i == 0 and success_rate < 0.15:
                print("\n❌ Phase 0 실패! 협력 학습이 제대로 되지 않았습니다.")
                break
            if i == 1 and success_rate < 0.1:
                print("\n⚠️ Phase 1 성공률 낮음. 그래도 Phase 2로 진행합니다.")

        except (KeyboardInterrupt, Exception):
            return

    print("\n" + "=" * 70)
    print("🎉 MAPPO Curriculum Learning 최종 요약")
    print("=" * 70)
    for name, rate in final_success_rates.items():
        print(f"{name:<35}: {rate*100:5.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()