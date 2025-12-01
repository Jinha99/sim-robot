"""
움직이는 장애물 모듈

로봇이 STAY 행동을 학습하기 위한 규칙적으로 움직이는 장애물 구현
"""
import random


class MovingObstacle:
    """
    규칙적으로 움직이는 장애물
    
    로봇이 STAY 행동으로 회피하는 것을 학습하기 위한 장애물
    로봇의 이동/회전 시간(5초/3초)과 유사하게 3초 또는 5초 간격으로 이동
    """
    
    def __init__(self, pattern="horizontal", start_pos=(0, 0), move_range=(-2, 2), speed=1, move_interval=None):
        """
        Args:
            pattern: 움직임 패턴 ("horizontal", "vertical")
            start_pos: 시작 위치 (x, y)
            move_range: 이동 범위 (min, max)
            speed: 이동 속도 (칸 수)
            move_interval: 이동 간격 (스텝 수). None이면 3 또는 5 중 랜덤 선택
        """
        self.pattern = pattern
        self.start_pos = start_pos
        self.move_range = move_range
        self.speed = speed
        
        # 이동 간격: 3초 또는 5초에 해당하는 스텝 수
        # 로봇: 이동 5초, 회전 3초 → 장애물도 유사한 간격
        if move_interval is None:
            self.move_interval = random.choice([3, 5])
        else:
            self.move_interval = move_interval
        
        # 현재 위치
        self.x, self.y = start_pos
        
        # 이동 방향 (1: 증가, -1: 감소)
        self.direction = 1
        
        # 스텝 카운터
        self._step_counter = 0
        
    def step(self):
        """한 스텝 진행: move_interval마다 장애물 이동"""
        self._step_counter += 1
        
        # move_interval마다 이동
        if self._step_counter >= self.move_interval:
            self._step_counter = 0
            
            # 새로운 이동 간격 랜덤 선택 (3초 또는 5초)
            self.move_interval = random.choice([3, 5])
            
            if self.pattern == "horizontal":
                # 수평 왕복 이동
                self.x += self.direction * self.speed
                
                # 범위 체크 및 방향 전환
                if self.x >= self.move_range[1]:
                    self.x = self.move_range[1]
                    self.direction = -1
                elif self.x <= self.move_range[0]:
                    self.x = self.move_range[0]
                    self.direction = 1
                    
            elif self.pattern == "vertical":
                # 수직 왕복 이동
                self.y += self.direction * self.speed
                
                # 범위 체크 및 방향 전환
                if self.y >= self.move_range[1]:
                    self.y = self.move_range[1]
                    self.direction = -1
                elif self.y <= self.move_range[0]:
                    self.y = self.move_range[0]
                    self.direction = 1
    
    def get_position(self):
        """현재 위치 반환"""
        return (self.x, self.y)
    
    def reset(self):
        """초기 위치로 리셋"""
        self.x, self.y = self.start_pos
        self.direction = 1
        self._step_counter = 0
        self.move_interval = random.choice([3, 5])
    
    def __repr__(self):
        return f"MovingObstacle(pos={self.get_position()}, pattern={self.pattern}, dir={self.direction}, interval={self.move_interval})"


def create_horizontal_obstacle(y=0, speed=1, move_interval=None):
    """
    수평 왕복 장애물 생성 (헬퍼 함수)
    
    Args:
        y: y 좌표 (기본값: 0, 중앙 라인)
        speed: 이동 속도
        move_interval: 이동 간격 (None이면 3/5 랜덤)
    
    Returns:
        MovingObstacle 인스턴스
    """
    return MovingObstacle(
        pattern="horizontal",
        start_pos=(-2, y),  # 왼쪽 끝에서 시작
        move_range=(-2, 2),
        speed=speed,
        move_interval=move_interval
    )


def create_vertical_obstacle(x=0, speed=1, move_interval=None):
    """
    수직 왕복 장애물 생성 (헬퍼 함수)
    
    Args:
        x: x 좌표 (기본값: 0, 중앙 라인)
        speed: 이동 속도
        move_interval: 이동 간격 (None이면 3/5 랜덤)
    
    Returns:
        MovingObstacle 인스턴스
    """
    return MovingObstacle(
        pattern="vertical",
        start_pos=(x, -2),  # 아래쪽 끝에서 시작
        move_range=(-2, 2),
        speed=speed,
        move_interval=move_interval
    )


def create_moving_obstacles(count=1):
    """
    다양한 조합의 움직이는 장애물 생성
    
    목표 위치를 피하도록 설계:
    - Tail 목표: (0, 0)
    - Head 목표: (1, 0), (-1, 0), (0, 1), (0, -1)
    
    Args:
        count: 생성할 장애물 개수 (1-3)
    
    Returns:
        MovingObstacle 리스트
    """
    obstacles = []
    
    if count >= 1:
        # 첫 번째: 수평 아래쪽 (y=-2)
        # 목표 (0,0), (0,1), (0,-1) 모두 피함 ✓
        obstacles.append(create_horizontal_obstacle(y=-2, speed=1))
    
    if count >= 2:
        # 두 번째: 수직 오른쪽 (x=2)
        # 목표 (0,0), (1,0), (-1,0) 모두 피함 ✓
        obstacles.append(create_vertical_obstacle(x=2, speed=1))
    
    if count >= 3:
        # 세 번째: 수평 위쪽 (y=2)
        # 목표 (0,0), (0,1), (0,-1) 모두 피함 ✓
        obstacles.append(create_horizontal_obstacle(y=2, speed=1))
    
    return obstacles

