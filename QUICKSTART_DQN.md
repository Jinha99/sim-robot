# DQN 학습 빠른 시작 가이드

## 📦 필수 패키지 설치

**중요: macOS에서는 `pip3` 또는 `python3.11 -m pip` 사용!**

```bash
# 방법 1: pip3 사용
pip3 install -r requirements.txt

# 방법 2: python3.11 -m pip 사용
python3.11 -m pip install -r requirements.txt

# 방법 3: 개별 설치
pip3 install torch numpy
```

## 🚀 DQN 학습 실행

```bash
cd src
python3.11 train_dqn.py
```

## 📊 학습 과정

학습이 진행되면 다음과 같은 로그가 출력됩니다:

```
============================================================
DQN 학습 시작
============================================================
에피소드 수: 100
시뮬레이션 시간: 200초
배치 크기: 32
초기 Epsilon: 1.000
============================================================
Ep   10/100 | Reward:  -45.2 | Steps: 12.3 | Loss: 0.0234 | ε: 0.904 | Win:   0 | Fail:   8
Ep   20/100 | Reward:  -32.1 | Steps: 15.7 | Loss: 0.0198 | ε: 0.817 | Win:   2 | Fail:  15
...
```

## 🎯 하이퍼파라미터 조정

`src/train_dqn.py` 파일에서 수정 가능:

```python
# DQN 에이전트 파라미터
agent = DQNAgent(
    state_dim=13,           # 상태 차원 (고정)
    action_dim=3,           # 행동 개수 (고정)
    learning_rate=0.001,    # 학습률 (↓ 더 안정적, ↑ 더 빠름)
    gamma=0.99,             # 할인율 (미래 보상 가중치)
    epsilon_start=1.0,      # 초기 탐험 확률
    epsilon_end=0.01,       # 최소 탐험 확률
    epsilon_decay=0.995,    # 탐험 감소율
)

# 트레이너 파라미터
trainer = DQNTrainer(
    num_episodes=100,       # 학습 에피소드 수 (↑ 더 오래 학습)
    termination_time=200,   # 에피소드당 최대 시간 (초)
    batch_size=32,          # 배치 크기
    buffer_size=10000,      # Replay Buffer 크기
    log_interval=10,        # 로그 출력 간격
    save_interval=50,       # 모델 저장 간격
)
```

## 💾 학습된 모델 사용

학습된 모델은 `outputs/dqn_worm_robot.pth`에 저장됩니다.

모델을 로드하여 사용:

```python
from rl.agent import DQNAgent

# 에이전트 생성
agent = DQNAgent(state_dim=13, action_dim=3)

# 학습된 모델 로드
agent.load("models/dqn_worm_robot.pth")

# 평가 모드 (탐험 안 함)
agent.epsilon = 0.0

# 시뮬레이션 실행
from main import run_simulation
system = run_simulation(rl_agent=agent, verbose=True)
```

## 🔍 현재 구현 상태

### ✅ 구현 완료

- DQN 에이전트 (Q-Network)
- Experience Replay Buffer
- Epsilon-greedy 탐험 전략
- Controller 연동
- 보상 함수
- 학습 루프
- 모델 저장/로드

### ⚠️ 간소화된 부분 (향후 개선 가능)

- 스텝별 경험 수집 대신 에피소드 기반 학습
- Target Network 미사용 (옵션으로 추가 가능)
- 행동 추적 간략화

### 🚧 향후 개선 가능

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- ~~멀티 에이전트 협력 학습 (MADDPG 등)~~ → **QMIX 또는 Independent DQN 권장** (이산 행동 공간에 적합)
- 학습 시각화 (TensorBoard)

### ⚠️ 학습이 안될 때

**문제**: 로봇 4개가 모두 성공해야 하므로 성공 경험이 없어 학습 진행 안됨

**해결책**: **Curriculum Learning 사용** (강력 추천!)

```bash
cd src
python3.11 train_curriculum.py
```

자세한 내용은 [CURRICULUM_LEARNING.md](../CURRICULUM_LEARNING.md) 참고

## 🐛 트러블슈팅

### pip 명령어가 없다고 나옴

```bash
# macOS에서는 pip3 사용
pip3 install torch numpy

# 또는
python3.11 -m pip install torch numpy
```

### PyTorch 설치 문제

```bash
# CPU 버전 (권장)
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA) 버전
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

### 학습이 너무 느림

- `num_episodes` 줄이기 (100 → 50)
- `termination_time` 줄이기 (200 → 100)

### 학습이 안됨 (보상이 안 올라감)

- `learning_rate` 조정 (0.001 → 0.0005)
- `epsilon_decay` 조정 (0.995 → 0.99, 더 빠른 탐험 감소)
- 더 많은 에피소드 학습 (100 → 500)

### ModuleNotFoundError 발생

```bash
# src 디렉토리에서 실행해야 함
cd src
python3.11 train_dqn.py
```

## 📚 참고 자료

- DQN 논문: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- PyTorch 튜토리얼: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
