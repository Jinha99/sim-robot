"""
Worm Robot Simulation - Environment Model
환경 DEVS 모델 정의
"""

from pypdevs.DEVS import AtomicDEVS
from pypdevs.infinity import INFINITY

from config import DIR_NAMES, STATUS_RUNNING, STATUS_WIN, STATUS_FAIL
from utils import in_bounds, get_sensor_area


# ========================================
# Environment 상태 클래스
# ========================================

class EnvironmentState:
    """환경의 내부 상태를 표현하는 클래스"""

    def __init__(self):
        self.robot_positions = {}  # {robot_id: {"head": (x,y), "tail": (x,y), "direction": int}}
        self.status = STATUS_RUNNING
        self.step_count = 0
        self.phase = "INIT"        # 상태: INIT, IDLE, PROCESSING
        self.pending_updates = []  # 대기 중인 로봇 업데이트
        #self.rewards = {}          # {robot_id: reward} - 각 로봇의 보상
        #self.prev_distances = {}   # {robot_id: distance} - 이전 스텝의 목표까지 거리

    def __str__(self):
        return (
            f"Environment["
            f"상태:{self.phase},"
            f"게임상태:{self.status},"
            f"스텝:{self.step_count},"
            f"로봇수:{len(self.robot_positions)}]"
        )


# ========================================
# Environment 모델 (Atomic DEVS)
# ========================================

class Environment(AtomicDEVS):
    """로봇들이 활동하는 환경의 DEVS 모델"""

    def __init__(self, num_robots=1, initial_positions=None, robot_goals=None, obstacles=None, moving_obstacles=None):
        """
        Args:
            num_robots: 로봇 수
            initial_positions: 초기 로봇 위치 리스트
            robot_goals: 각 로봇의 목적지 딕셔너리 {robot_id: goal_position}
            obstacles: 정적 장애물 위치 리스트 [(x, y), ...] (선택)
            moving_obstacles: 움직이는 장애물 리스트 [MovingObstacle, ...] (선택)
        """
        AtomicDEVS.__init__(self, "Environment")
        self.num_robots = num_robots
        self.state = EnvironmentState()
        self.initial_positions = initial_positions or []
        self.robot_goals = robot_goals or {}  # 각 로봇의 목적지
        self.obstacles = obstacles or []  # 정적 장애물 위치
        self.moving_obstacles = moving_obstacles or []  # 움직이는 장애물

        # 로봇 초기 위치 설정
        for pos_data in self.initial_positions:
            self.state.robot_positions[pos_data["id"]] = {
                "head": pos_data["head"],
                "tail": pos_data["tail"],
                "direction": pos_data["dir"]
            }

        # 입력 포트 (로봇들로부터)
        self.robot_done_in = [self.addInPort(f"robot{i}_done_in") for i in range(num_robots)]

        # 출력 포트 (컨트롤러로)
        self.obs_out = self.addOutPort("obs_out")          # 관찰 데이터
        self.status_out = self.addOutPort("status_out")    # 게임 상태

    def timeAdvance(self):
        """시간 진행 함수"""
        if self.state.phase == "INIT":
            return 0  # 즉시 초기화
        elif self.state.phase == "IDLE":
            if self.state.status != STATUS_RUNNING:
                return INFINITY  # 게임 종료
            return INFINITY  # 로봇 행동 대기
        elif self.state.phase == "PROCESSING":
            return 0  # 즉시 처리
        return INFINITY

    def intTransition(self):
        """내부 전이 함수"""
        if self.state.phase == "INIT":
            # 초기 관찰 데이터를 보낸 후 IDLE로 전환
            self.state.phase = "IDLE"
            return self.state

        elif self.state.phase == "PROCESSING":
            # 로봇 위치 업데이트 및 승패 판정
            self._update_environment()
            self.state.pending_updates = []
            self.state.phase = "IDLE"
            return self.state

        return self.state

    def extTransition(self, inputs):
        """외부 전이 함수 - 로봇 행동 완료 신호 수신"""
        if self.state.phase == "IDLE":
            # 로봇들의 행동 완료 신호 수집
            for i, port in enumerate(self.robot_done_in):
                update = inputs.get(port)
                if update:
                    self.state.pending_updates.append(update)

            # 모든 로봇이 완료했으면 처리 시작
            if len(self.state.pending_updates) > 0:
                self.state.phase = "PROCESSING"

        return self.state

    def outputFnc(self):
        """출력 함수 - 관찰 데이터 및 상태 전송"""
        # INIT이나 PROCESSING 상태에서만 출력 (IDLE에서는 출력 안함)
        if self.state.phase in ["INIT", "PROCESSING"]:
            obs = self._generate_observations()
            return {
                self.obs_out: obs,
                self.status_out: {
                    "status": self.state.status,
                    "step": self.state.step_count
                }
            }
        return {}

    def _update_environment(self):
        """환경 업데이트: 로봇 위치 갱신 및 승패 판정"""
        # 1. 움직이는 장애물 업데이트
        for moving_obs in self.moving_obstacles:
            moving_obs.step()
        
        # 2. 로봇 위치 갱신
        for update in self.state.pending_updates:
            rid = update["robot_id"]
            self.state.robot_positions[rid] = {
                "head": update["head"],
                "tail": update["tail"],
                "direction": update["direction"]
            }

        self.state.step_count += 1

        # 보상 계산 (승패 판정 전에 수행)
        # self._calculate_rewards()

        # 승패 판정
        if self._check_fail():
            self.state.status = STATUS_FAIL
        elif self._check_win():
            self.state.status = STATUS_WIN

    def _check_fail(self):
        """실패 조건 확인: 격자 이탈, 로봇 충돌, 또는 장애물 충돌"""
        positions = self.state.robot_positions

        # 격자 범위 이탈 확인
        for rid, pos_data in positions.items():
            if not in_bounds(pos_data["head"]) or not in_bounds(pos_data["tail"]):
                print(f"[실패] Robot {rid}가 격자를 벗어났습니다!")
                return True

        # 정적 장애물 충돌 확인
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]
            
            if head in self.obstacles:
                print(f"[실패] Robot {rid}의 앞발이 {head} 위치의 정적 장애물과 충돌!")
                return True
            
            if tail in self.obstacles:
                print(f"[실패] Robot {rid}의 뒷발이 {tail} 위치의 정적 장애물과 충돌!")
                return True
        
        # 움직이는 장애물 충돌 확인
        moving_obs_positions = [obs.get_position() for obs in self.moving_obstacles]
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]
            
            if head in moving_obs_positions:
                print(f"[실패] Robot {rid}의 앞발이 {head} 위치의 움직이는 장애물과 충돌!")
                return True
            
            if tail in moving_obs_positions:
                print(f"[실패] Robot {rid}의 뒷발이 {tail} 위치의 움직이는 장애물과 충돌!")
                return True

        # 로봇 간 충돌 확인
        occupied = {}
        for rid, pos_data in positions.items():
            head = pos_data["head"]
            tail = pos_data["tail"]

            # 앞발 충돌 체크
            if head in occupied:
                print(f"[실패] {head} 위치에서 Robot {rid}와 Robot {occupied[head]}가 충돌!")
                return True
            occupied[head] = rid

            # 뒷발 충돌 체크 (이제 (0,0) 예외 없음)
            if tail in occupied:
                print(f"[실패] {tail} 위치에서 Robot {rid}와 Robot {occupied[tail]}가 충돌!")
                return True
            occupied[tail] = rid

        return False

    def _check_win(self):
        """
        승리 조건 확인:
        - num_robots = 1: (0, 1) 한 칸만 차지 (원하면 사용)
        - num_robots = 2: (0, 1), (0, -1) 두 칸 모두 차지
        - num_robots = 3: 십자 모양 4칸 중 아무 3칸 차지
        - num_robots = 4: 십자 모양 4칸 모두 차지
        """
        num_robots = len(self.state.robot_positions)
        if num_robots == 0:
            return False

        heads = {
            pos_data["head"]
            for pos_data in self.state.robot_positions.values()
        }

        # 십자 전체 타겟
        full_targets = {(0, 1), (1, 0), (0, -1), (-1, 0)}

        if num_robots == 1:
            # 옵션: 1대만 있을 때는 (0,1)에 서 있으면 승리로 볼 수도 있음
            target_positions = {(0, 1)}
            if heads == target_positions:
                print("[승리] 1개 로봇이 중앙 위 칸에 배치되었습니다!")
                return True
            return False

        elif num_robots == 2:
            # 1차 목표: 위/아래 두 칸 정확히 차지
            target_positions = {(0, 1), (0, -1)}
            if heads == target_positions:
                print("[승리] 2개 로봇의 앞발이 위/아래 칸에 배치되었습니다!")
                return True
            return False

        else:
            # 3대 이상: 십자 4칸 중에서 로봇 수만큼 distinct 칸을 차지하면 OK
            # (3대면 아무 3칸, 4대면 4칸 모두)
            if heads.issubset(full_targets) and len(heads) == num_robots:
                print(f"[승리] {num_robots}개 로봇의 앞발이 중앙 십자 영역에 배치되었습니다!")
                return True
            return False

    def _generate_observations(self):
        """각 로봇의 센서 관찰 데이터 생성"""
        observations = {}

        for rid, pos_data in self.state.robot_positions.items():
            head = pos_data["head"]
            sensor_area = get_sensor_area(head)

            # 센서 범위 내 다른 로봇 감지
            detected = []
            for other_id, other_pos in self.state.robot_positions.items():
                if other_id == rid:
                    continue
                if other_pos["head"] in sensor_area or other_pos["tail"] in sensor_area:
                    detected.append({
                        "robot_id": other_id,
                        "head": other_pos["head"],
                        "tail": other_pos["tail"]
                    })

            # 움직이는 장애물 위치
            moving_obs_positions = [obs.get_position() for obs in self.moving_obstacles]
            
            observations[rid] = {
                "own_head": head,
                "own_tail": pos_data["tail"],
                "own_direction": pos_data["direction"],
                "detected_robots": detected,
                "goal_position": self.robot_goals.get(rid, (0, 0)),  # 목적지 좌표 전달
                "obstacles": self.obstacles,  # 정적 장애물 위치 전달
                "moving_obstacles": moving_obs_positions  # 움직이는 장애물 위치 전달
            }

        return observations
