# TensorBoard 사용 가이드 📊

TensorBoard를 사용하여 DQN 학습 과정을 실시간으로 모니터링할 수 있습니다!

## 📦 설치

```bash
pip3 install tensorboard
```

## 🚀 사용 방법

### 1. 학습 실행

TensorBoard는 자동으로 활성화됩니다:

```bash
cd src
python3.11 train_dqn.py
```

출력:

```
📊 TensorBoard 로깅 활성화: runs/worm_robot_dqn
   실행: tensorboard --logdir=runs
```

### 2. TensorBoard 실행

**새 터미널 창**을 열고:

```bash
# 프로젝트 루트 디렉토리에서
cd /Users/choo/PycharmProjects/worm-robot-project
tensorboard --logdir=runs
```

출력:

```
TensorBoard 2.x.x at http://localhost:6006/ (Press CTRL+C to quit)
```

### 3. 브라우저에서 확인

브라우저를 열고 접속:

```
http://localhost:6006
```

## 📊 볼 수 있는 그래프들

### 1. Reward/episode

- 각 에피소드의 총 보상
- **상승하면** 학습이 잘 되고 있는 것!

### 2. Steps/episode

- 각 에피소드의 스텝 수
- 목표에 빠르게 도달하면 줄어듦

### 3. Loss/episode

- 학습 손실
- **감소하면** 네트워크가 안정화되는 것

### 4. Epsilon

- 탐험 확률 (1.0 → 0.01)
- **감소하면** 랜덤에서 학습된 행동으로 전환

### 5. Success/total & Fail/total

- 누적 성공/실패 횟수
- 성공이 증가하는지 확인

### 6. Result/win & Result/fail

- 각 에피소드의 성공/실패 (0 또는 1)
- 패턴 확인 가능

## 🎨 TensorBoard 사용 팁

### 스무딩 조절

우측 상단의 **Smoothing** 슬라이더로 그래프를 부드럽게 표시:

- 0.0: 원본 데이터 (울퉁불퉁)
- 0.6: 적당히 부드럽게 (추천)
- 0.99: 매우 부드럽게 (트렌드만)

### 여러 실험 비교

```bash
# 실험 1: Learning Rate 0.001
python3.11 train_dqn.py  # runs/worm_robot_dqn_1

# 실험 2: Learning Rate 0.0005
# train_dqn.py에서 tensorboard_dir 수정
# runs/worm_robot_dqn_2

# TensorBoard에서 둘 다 보기
tensorboard --logdir=runs
```

### 그래프 다운로드

그래프 위에 마우스를 올리면 다운로드 버튼:

- CSV: 데이터
- SVG: 이미지
- JSON: 원본 데이터

## 🔧 문제 해결

### "TensorBoard를 사용하려면 설치하세요" 메시지

```bash
pip3 install tensorboard
```

### TensorBoard 비활성화하고 싶을 때

`train_dqn.py`에서 트레이너 생성 시:

```python
trainer = DQNTrainer(
    agent=agent,
    create_system_fn=create_system,
    use_tensorboard=False,  # ← 이거 추가
    ...
)
```

### 포트 6006이 이미 사용 중

다른 포트로 실행:

```bash
tensorboard --logdir=runs --port=6007
```

### 이전 로그 삭제

```bash
rm -rf runs/
```

## 📱 실시간 모니터링

학습 중에 TensorBoard를 열어두면:

- 새로고침 버튼 클릭 또는
- 자동 새로고침 활성화
- 실시간으로 학습 과정 관찰!

## 🎯 학습 진단

### 학습이 잘 되고 있는지 확인:

✅ **좋은 신호:**

- Reward가 상승 추세
- Loss가 감소 후 안정화
- Success가 증가
- Epsilon이 감소하며 성공률 증가

❌ **나쁜 신호:**

- Reward가 계속 하락
- Loss가 폭발 (너무 큼)
- Success가 0으로 계속
- Epsilon이 낮아져도 성공 못함

### 문제별 대응:

**Loss가 폭발:**

- Learning rate 낮추기 (0.001 → 0.0005)

**Reward가 안 올라감:**

- 더 많은 에피소드 학습
- Epsilon decay 조정 (0.995 → 0.99)
- 보상 함수 재설계

**성공률이 0:**

- 시뮬레이션 시간 늘리기
- 목표 조건 확인
- 초기 위치 확인

## 💡 예시 학습 곡선

정상적인 학습 과정:

```
Reward
 │
50│                          ┌────
 0│                     ┌───┘
-50│                ┌───┘
-100│───────────────┘
    └────────────────────────── Episode
     0   20   40   60   80  100

Loss
 │
2│──┐
1│  └──┐
0│     └────────────────
  └────────────────────────── Episode
   0   20   40   60   80  100

Success Rate (누적)
 │
50│                    ┌────
25│               ┌───┘
 0│───────────────┘
  └────────────────────────── Episode
   0   20   40   60   80  100
```

## 🎬 시작하기

```bash
# 1. TensorBoard 설치
pip3 install tensorboard

# 2. 학습 시작 (첫 번째 터미널)
cd src
python3.11 train_dqn.py

# 3. TensorBoard 실행 (두 번째 터미널)
cd /Users/choo/PycharmProjects/worm-robot-project
tensorboard --logdir=runs

# 4. 브라우저 열기
# http://localhost:6006
```

즐거운 학습 모니터링 되세요! 🚀
