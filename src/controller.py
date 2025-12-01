"""
Worm Robot Simulation - Controller Model
컨트롤러 DEVS 모델 정의 (강화학습 연동 지점)
"""

import random
from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import (
    STATUS_RUNNING,
    ACTION_MOVE,
    ACTION_ROTATE_CW,
    ACTION_ROTATE_CCW,
    ACTION_STAY,
)


# ========================================
# Controller 상태 클래스
# ========================================

class ControllerState:
    """컨트롤러의 내부 상태를 표현하는 클래스"""

    def __init__(self):
        self.observations = {}
        self.status = STATUS_RUNNING
        self.step = 0
        self.phase = "IDLE"  # 상태: IDLE, DECIDING

    def __str__(self):
        return (
            f"Controller["
            f"상태:{self.phase},"
            f"스텝:{self.step},"
            f"게임상태:{self.status}]"
        )


# ========================================
# Controller 모델 (Atomic DEVS)
# ========================================

class Controller(AtomicDEVS):
    """
    로봇들의 행동을 결정하는 컨트롤러 DEVS 모델

    강화학습 연동 지점:
    - _select_action() 메서드를 수정하여 RL 에이전트 통합 가능
    """

    def __init__(self, num_robots=1, rl_agent=None):
        """
        Args:
            num_robots: 로봇 수
            rl_agent: (선택) 강화학습 에이전트 인스턴스
        """
        AtomicDEVS.__init__(self, "Controller")
        self.num_robots = num_robots
        self.state = ControllerState()
        self.rl_agent = rl_agent  # 강화학습 에이전트 (None이면 휴리스틱 사용)

        # 입력 포트
        self.obs_in = self.addInPort("obs_in")          # 관찰 데이터
        self.status_in = self.addInPort("status_in")    # 게임 상태

        # 출력 포트 (로봇들로)
        self.action_out = [self.addOutPort(f"action{i}_out") for i in range(num_robots)]

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "IDLE":
            return INFINITY  # 관찰 데이터 대기
        elif self.state.phase == "DECIDING":
            return 0  # 즉시 행동 결정
        return INFINITY

    def intTransition(self):
        """내부 전이 함수 - 행동 결정 완료"""
        if self.state.phase == "DECIDING":
            self.state.phase = "IDLE"
        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 관찰 데이터 수신"""
        # 관찰 데이터 수신
        obs = inputs.get(self.obs_in)
        if obs:
            self.state.observations = obs

        # 게임 상태 수신
        status = inputs.get(self.status_in)
        if status:
            self.state.status = status["status"]
            self.state.step = status["step"]

        # 게임이 진행 중이고 관찰 데이터가 있으면 결정 시작
        if self.state.observations and self.state.status == STATUS_RUNNING:
            self.state.phase = "DECIDING"

        return self.state

    def outputFnc(self):
        """출력 함수 - 각 로봇에 행동 명령 전송"""
        if self.state.phase == "DECIDING":
            actions = {}
            for rid in range(self.num_robots):
                if rid in self.state.observations:
                    action = self._select_action(rid, self.state.observations[rid])
                    actions[self.action_out[rid]] = action
            return actions
        return {}

    def _select_action(self, rid, obs):
        """
        rid: 로봇 ID
        obs: 해당 로봇의 관측 딕셔너리
        """
        if self.rl_agent is not None:
            # ============ 1) 주변 로봇 거리 기반 위험 감지 ============
            own_head = obs["own_head"]              # 내 앞발 좌표 (x, y)
            detected = obs["detected_robots"]       # 센서에 잡힌 다른 로봇들 리스트

            danger = False
            for robot in detected:
                # head / tail 둘 다 검사
                for key in ("head", "tail"):
                    other = robot[key]              # (x, y)
                    dx = abs(other[0] - own_head[0])
                    dy = abs(other[1] - own_head[1])
                    # 상/하/좌/우/대각선 1칸 이내 → max(|dx|, |dy|) <= 1
                    if max(dx, dy) <= 1:
                        danger = True
                        break
                if danger:
                    break

            # 🔒 안전 룰: 주변 1칸 안에 다른 로봇 있으면 무조건 STAY
            if danger:
                return {"type": ACTION_STAY}

            # ============ 2) 안전할 때만 RL에게 맡김 ============
            state = self._observation_to_state(obs)
            action_idx = self.rl_agent.get_action(state, training=True)

            # MAPPO 쪽이 action_dim=4 이므로 4개 모두 매핑
            action_types = [
                ACTION_MOVE,
                ACTION_ROTATE_CW,
                ACTION_ROTATE_CCW,
                ACTION_STAY,   # 4번째 액션은 자발적 STAY
            ]

            # 방어 코드: 범위를 벗어나면 STAY
            if not (0 <= action_idx < len(action_types)):
                return {"type": ACTION_STAY}

            return {"type": action_types[action_idx]}


    def _observation_to_state(self, obs):
        """
        관찰 데이터를 RL 에이전트가 사용할 상태 표현으로 변환

        Args:
            obs: 관찰 데이터

        Returns:
            강화학습 상태 표현 (numpy array)
        """
        import numpy as np
        
        # 자신의 위치 (정규화: -3~3 → -1~1)
        own_head = obs["own_head"]
        own_tail = obs["own_tail"]
        
        # 목표 위치
        goal_position = obs["goal_position"]
        
        # 목표까지 벡터 계산
        vector_to_goal_head = (goal_position[0] - own_head[0], goal_position[1] - own_head[1])
        vector_to_goal_tail = (0 - own_tail[0], 0 - own_tail[1])  # 뒷발은 항상 (0,0)
        
        # 방향 (0~3)
        direction = obs["own_direction"]
        
        # 주변 로봇 정보 (간단하게: 개수와 가장 가까운 로봇까지 거리)
        detected = obs["detected_robots"]
        num_nearby = len(detected)
        
        closest_dist = 10.0  # 기본값 (멀리 있음)
        if detected:
            for robot in detected:
                dist = abs(robot["head"][0] - own_head[0]) + abs(robot["head"][1] - own_head[1])
                closest_dist = min(closest_dist, dist)
        
        # 상태 벡터 구성 (13차원)
        state = np.array([
            own_head[0] / 3.0,          # -1 ~ 1
            own_head[1] / 3.0,          # -1 ~ 1
            own_tail[0] / 3.0,          # -1 ~ 1
            own_tail[1] / 3.0,          # -1 ~ 1
            direction / 3.0,            # 0 ~ 1
            vector_to_goal_head[0] / 6.0,  # -1 ~ 1
            vector_to_goal_head[1] / 6.0,  # -1 ~ 1
            vector_to_goal_tail[0] / 6.0,  # -1 ~ 1
            vector_to_goal_tail[1] / 6.0,  # -1 ~ 1
            goal_position[0] / 3.0,     # -1 ~ 1
            goal_position[1] / 3.0,     # -1 ~ 1
            num_nearby / 3.0,           # 0 ~ 1
            closest_dist / 10.0         # 0 ~ 1
        ], dtype=np.float32)
        
        return state
