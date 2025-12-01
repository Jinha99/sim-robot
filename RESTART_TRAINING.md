# 🔄 학습 재시작 가이드

## 문제 상황

37970 에피소드 동안 성공 0번 → 학습이 전혀 안 됨

## ✅ 개선 사항

### 1. 보상 함수 대폭 개선

**이전:**

- 희소한 보상 (거의 학습 신호 없음)
- 성공해야만 +100점

**개선:**

- **거리 기반 보상**: 가까울수록 높은 점수 (0~120점)
- **거리 감소 보너스**: 가까워지면 즉시 보상 (+5배)
- **부분 성공 보너스**:
  - 뒷발 도착: +50점
  - 앞발 도착: +50점
  - 완전 성공: +300점
- **촘촘한 피드백**: 매 스텝마다 학습 신호

### 2. 학습 파라미터 최적화

- Learning rate: 0.001 → **0.0005** (더 안정적)
- Epsilon end: 0.01 → **0.05** (더 많은 탐험)
- Epsilon decay: 0.995 → **0.998** (더 천천히 감소)
- Termination time: 200 → **100초** (더 빠른 에피소드)

### 3. 에피소드 수 조정

- 500000000 → **1000** (합리적인 수준)

## 🚀 재시작 방법

### 1단계: 이전 모델 삭제 (중요!)

```bash
# 잘못 학습된 모델 삭제
rm models/dqn_worm_robot.pth

# TensorBoard 로그도 삭제 (선택)
rm -rf runs/
```

### 2단계: 새로 학습 시작

```bash
cd src
python3.11 train_dqn.py
```

### 3단계: TensorBoard로 모니터링

**새 터미널에서:**

```bash
tensorboard --logdir=runs
```

**브라우저:** http://localhost:6006

## 📊 예상 결과

### 이전 (37970 에피소드)

```
Reward: -120 ~ -80 (불규칙)
Success: 0
Loss: 불안정
```

### 개선 후 (예상)

```
에피소드 100:  Reward: ~50  (거리가 줄어듦)
에피소드 300:  Reward: ~80  (뒷발 도착 시작)
에피소드 500:  Reward: ~150 (앞발도 도착)
에피소드 800:  Reward: ~300 (완전 성공!)
```

## 🎯 학습 신호 확인

TensorBoard에서 확인할 것:

### ✅ 좋은 신호

- **Reward가 상승** (0 → 100 → 200 → 300)
- **Loss가 감소 후 안정화**
- **Success가 서서히 증가**

### ❌ 여전히 안 되면

1. **에피소드 100~200 확인**

   - Reward가 조금이라도 증가하는가?
   - 아니면 여전히 -100 근처인가?

2. **디버깅 모드**

   ```python
   # train_dqn.py에서
   system = run_simulation(rl_agent=agent, verbose=True)
   ```

   - 로봇이 실제로 움직이는지 확인

3. **더 쉬운 목표 설정**
   - 완전 성공 대신 "뒷발만 (0,0)에" 같이 목표 간소화

## 💡 왜 이전엔 안 됐나?

### 문제 1: 희소 보상 (Sparse Reward)

```python
# 이전
성공: +100
실패: -50
그 외: ~0

→ 37970번 실패 → 학습 불가능
```

### 문제 2: 학습 신호 부족

```python
# 이전
"성공하지 않는 한 뭐가 좋은지 모름"

# 개선
"조금이라도 가까워지면 보상!"
→ 학습 가능
```

## 🔧 추가 개선 (선택)

### 옵션 1: Curriculum Learning

먼저 쉬운 문제부터:

```python
# config.py에서 격자 크기 축소
GRID_SIZE = 5  # 7 → 5 (더 쉬움)
```

### 옵션 2: 더 긴 학습

```python
# train_dqn.py
num_episodes=2000  # 1000 → 2000
```

### 옵션 3: 학습률 더 낮추기

```python
# train_dqn.py
learning_rate=0.0003  # 0.0005 → 0.0003
```

## 📝 체크리스트

재시작 전 확인:

- [ ] 이전 모델 삭제 (`rm models/dqn_worm_robot.pth`)
- [ ] TensorBoard 준비
- [ ] 에피소드 수 확인 (1000)
- [ ] 시간 여유 확보 (1~2시간)

## 🎬 시작!

```bash
# 1. 모델 삭제
rm models/dqn_worm_robot.pth

# 2. 학습 시작
cd src
python3.11 train_dqn.py

# 3. TensorBoard (새 터미널)
tensorboard --logdir=runs
```

이번엔 잘 될 거예요! 🚀
