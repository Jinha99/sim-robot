"""
Worm Robot Simulation - System Model
전체 시스템 통합 DEVS 모델
"""

from pypdevs.DEVS import CoupledDEVS

import config
from robot import Robot
from environment import Environment
from controller import Controller


# ========================================
# 결합 모델: Worm Robot System
# ========================================

class WormRobotSystem(CoupledDEVS):
    """
    2관절 로봇 시스템 전체를 통합하는 결합 DEVS 모델

    구성 요소:
    - Robot: 개별 로봇 모델 (num_robots개)
    - Environment: 환경 모델 (1개)
    - Controller: 컨트롤러 모델 (1개)
    """

    def __init__(self, rl_agent=None, num_robots=None, obstacles=None, moving_obstacles=None):
        """
        Args:
            rl_agent: (선택) 강화학습 에이전트 인스턴스
            num_robots: (선택) 로봇 수 (None이면 config.NUM_ROBOTS 사용)
            obstacles: (선택) 정적 장애물 위치 리스트 [(x, y), ...]
            moving_obstacles: (선택) 움직이는 장애물 리스트 [MovingObstacle, ...]
        """
        CoupledDEVS.__init__(self, "WormRobotSystem")

        # 로봇 수 결정 (Curriculum Learning 지원)
        if num_robots is None:
            num_robots = config.NUM_ROBOTS
        
        # 매 시뮬레이션마다 새로운 랜덤 위치 생성 (장애물 위치 고려)
        robot_configs = config.generate_random_robot_configs(
            num_robots=num_robots,
            obstacles=obstacles,
            moving_obstacles=moving_obstacles
        )

        # 로봇 생성
        self.robots = []
        for config_item in robot_configs:
            robot = Robot(
                robot_id=config_item["id"],
                initial_head=config_item["head"],
                initial_tail=config_item["tail"],
                initial_direction=config_item["dir"]
            )
            self.robots.append(self.addSubModel(robot))

        # 환경 생성 (목적지 정보 및 장애물 포함)
        robot_goals = {config_item["id"]: config_item["goal"] for config_item in robot_configs}
        self.environment = self.addSubModel(
            Environment(
                num_robots=num_robots, 
                initial_positions=robot_configs, 
                robot_goals=robot_goals,
                obstacles=obstacles,
                moving_obstacles=moving_obstacles
            )
        )

        # 컨트롤러 생성
        self.controller = self.addSubModel(
            Controller(num_robots=num_robots, rl_agent=rl_agent)
        )

        # 포트 연결: 로봇 -> 환경
        for i, robot in enumerate(self.robots):
            self.connectPorts(robot.action_done_out, self.environment.robot_done_in[i])

        # 포트 연결: 환경 -> 컨트롤러
        self.connectPorts(self.environment.obs_out, self.controller.obs_in)
        self.connectPorts(self.environment.status_out, self.controller.status_in)

        # 포트 연결: 컨트롤러 -> 로봇
        for i, robot in enumerate(self.robots):
            self.connectPorts(self.controller.action_out[i], robot.action_in)

    def select(self, imm):
        """
        동시 발생 이벤트 우선순위 결정

        우선순위: 환경 > 컨트롤러 > 로봇

        Args:
            imm: 동시에 발생한 이벤트 리스트

        Returns:
            우선순위가 가장 높은 모델
        """
        if self.environment in imm:
            return self.environment
        if self.controller in imm:
            return self.controller
        return imm[0]
    
    # ========================================
    # RL 학습용 Step-by-Step 인터페이스
    # ========================================
    
    def get_observations(self):
        """현재 관찰 데이터 반환 (RL 학습용)"""
        return self.environment._generate_observations()
    
    def get_state_for_robot(self, robot_id):
        """특정 로봇의 상태 벡터 반환"""
        obs = self.environment._generate_observations()[robot_id]
        return self.controller._observation_to_state(obs)
    
    def step(self, actions):
        """
        한 스텝 실행 (RL 학습용) with Invalid Action Masking & Collision Detection

        Args:
            actions: {robot_id: action_idx} - 각 로봇의 행동

        Returns:
            observations: 다음 관찰
            rewards: {robot_id: reward} - 각 로봇의 보상 (충돌 패널티 포함)
            done: 에피소드 종료 여부
            status: 게임 상태 (STATUS_WIN, STATUS_FAIL, STATUS_RUNNING)
        """
        import random
        from collections import defaultdict

        # 각 로봇의 의도된 이동 계산
        intended_updates = []
        original_positions = {}

        for robot_id, action_idx in actions.items():
            robot_model = self.robots[robot_id]
            current_pos = self.environment.state.robot_positions[robot_id]

            # 현재 상태 저장
            direction = current_pos["direction"]
            head = current_pos["head"]
            tail = current_pos["tail"]
            original_positions[robot_id] = {"head": head, "tail": tail, "direction": direction}

            # Invalid Action Masking: 유효한 action인지 체크
            valid_actions = self._get_valid_actions(robot_id, head, tail, direction)

            if action_idx not in valid_actions:
                # Invalid action이면 랜덤하게 valid action 선택
                action_idx = random.choice(valid_actions) if valid_actions else 1  # 회전은 항상 안전

            # Action에 따라 새로운 위치 계산
            if action_idx == 0:  # FORWARD
                new_direction = direction
                delta = config.DIRECTIONS[new_direction]
                new_head = (head[0] + delta[0], head[1] + delta[1])
                new_tail = head
            elif action_idx == 1:  # TURN_LEFT
                new_direction = (direction - 1) % 4
                new_head = head
                new_tail = tail
            elif action_idx == 2:  # TURN_RIGHT
                new_direction = (direction + 1) % 4
                new_head = head
                new_tail = tail
            elif action_idx == 3:  # STAY (대기)
                new_direction = direction
                new_head = head
                new_tail = tail
            else:
                # 기본값 (안전장치)
                new_direction = direction
                new_head = head
                new_tail = tail

            intended_updates.append({
                "robot_id": robot_id,
                "head": new_head,
                "tail": new_tail,
                "direction": new_direction,
                "action_idx": action_idx
            })

        # 🔍 동일 칸 충돌 감지
        collision_penalties = {}
        head_positions = defaultdict(list)  # {position: [robot_ids]}
        tail_positions = defaultdict(list)

        # 각 로봇이 이동하려는 위치 수집
        for update in intended_updates:
            robot_id = update["robot_id"]
            head_positions[update["head"]].append(robot_id)
            tail_positions[update["tail"]].append(robot_id)

        # 충돌 감지: head끼리, head-tail, tail끼리 모두 체크
        blocked_robots = set()

        # 1. Head 위치 충돌 검사
        for pos, robot_ids in head_positions.items():
            if len(robot_ids) > 1:
                # 여러 로봇이 같은 head 위치로 이동하려 함
                blocked_robots.update(robot_ids)
                for rid in robot_ids:
                    collision_penalties[rid] = -50.0  # 충돌 패널티

        # 2. Head-Tail 교차 충돌 검사
        for pos in head_positions.keys():
            if pos in tail_positions and len(head_positions[pos]) > 0 and len(tail_positions[pos]) > 0:
                # 한 로봇의 head와 다른 로봇의 tail이 같은 위치
                head_robots = head_positions[pos]
                tail_robots = tail_positions[pos]
                # 다른 로봇끼리만 충돌로 판정
                for hrid in head_robots:
                    for trid in tail_robots:
                        if hrid != trid:
                            blocked_robots.add(hrid)
                            blocked_robots.add(trid)
                            collision_penalties[hrid] = -50.0
                            collision_penalties[trid] = -50.0

        # 3. Tail 위치 충돌 검사
        for pos, robot_ids in tail_positions.items():
            if len(robot_ids) > 1:
                # 여러 로봇이 같은 tail 위치로 이동하려 함
                blocked_robots.update(robot_ids)
                for rid in robot_ids:
                    collision_penalties[rid] = -50.0  # 충돌 패널티

        # 🚫 충돌한 로봇은 이동 취소 (원래 위치 유지)
        final_updates = []
        for update in intended_updates:
            robot_id = update["robot_id"]
            if robot_id in blocked_robots:
                # 충돌! 원래 위치로 되돌림
                original = original_positions[robot_id]
                final_updates.append({
                    "robot_id": robot_id,
                    "head": original["head"],
                    "tail": original["tail"],
                    "direction": original["direction"]
                })
            else:
                # 충돌 없음, 정상 이동
                final_updates.append({
                    "robot_id": robot_id,
                    "head": update["head"],
                    "tail": update["tail"],
                    "direction": update["direction"]
                })

        # Environment 업데이트
        self.environment.state.pending_updates = final_updates
        self.environment._update_environment()

        # 결과 수집
        observations = self.environment._generate_observations()
        status = self.environment.state.status
        done = (status != config.STATUS_RUNNING)

        # 충돌 패널티를 포함한 rewards 반환
        rewards = {
            rid: collision_penalties.get(rid, 0.0)
            for rid in self.environment.state.robot_positions.keys()
        }

        return observations, rewards, done, status
    
    def _get_valid_actions(self, robot_id, head, tail, direction):
        """
        특정 로봇이 선택할 수 있는 유효한 action들 반환
        
        Args:
            robot_id: 로봇 ID
            head: 앞발 위치
            tail: 뒷발 위치
            direction: 현재 방향
        
        Returns:
            list: 유효한 action 인덱스 리스트
        """
        valid_actions = []
        
        # 0: FORWARD - 격자 이탈과 장애물 체크
        delta = config.DIRECTIONS[direction]
        new_head = (head[0] + delta[0], head[1] + delta[1])
        new_tail = head
        
        if self._is_position_valid(new_head, new_tail):
            valid_actions.append(0)
        
        # 1: TURN_LEFT - 회전은 항상 안전
        valid_actions.append(1)
        
        # 2: TURN_RIGHT - 회전은 항상 안전
        valid_actions.append(2)
        
        # 3: STAY - 다중 로봇에서만 의미있음
        # 로봇이 2개 이상일 때만 STAY 허용 (정적 장애물에서는 무의미)
        if len(self.robots) >= 2:
            valid_actions.append(3)
        
        return valid_actions
    
    def _is_position_valid(self, head, tail):
        """
        위치가 유효한지 체크 (격자 내부 + 정적/동적 장애물 없음)
        
        Args:
            head: 앞발 위치
            tail: 뒷발 위치
        
        Returns:
            bool: 유효하면 True
        """
        # 격자 범위 체크 (-3 ~ 3)
        if not (-3 <= head[0] <= 3 and -3 <= head[1] <= 3):
            return False
        if not (-3 <= tail[0] <= 3 and -3 <= tail[1] <= 3):
            return False
        
        # 정적 장애물 체크
        obstacles = self.environment.obstacles
        if head in obstacles or tail in obstacles:
            return False
        
        # 움직이는 장애물 체크
        moving_obs_positions = [obs.get_position() for obs in self.environment.moving_obstacles]
        if head in moving_obs_positions or tail in moving_obs_positions:
            return False
        
        return True
    
    def is_done(self):
        """에피소드 종료 여부"""
        return self.environment.state.status != config.STATUS_RUNNING
    
    def get_status(self):
        """현재 게임 상태"""
        return self.environment.state.status
    
    def get_step_count(self):
        """현재 스텝 수"""
        return self.environment.state.step_count
