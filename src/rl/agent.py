"""
Worm Robot Simulation - DQN Agent
간단한 DQN 구현
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Q-Network (신경망)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    간단한 DQN 에이전트
    
    특징:
    - Epsilon-greedy 탐험
    - Experience Replay
    - Target Network (선택적)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        use_target_net=False,
        device="cpu"
    ):
        """
        Args:
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원 (3: 전진, 시계, 반시계)
            learning_rate: 학습률
            gamma: 할인율
            epsilon_start: 초기 탐험 확률
            epsilon_end: 최소 탐험 확률
            epsilon_decay: 탐험 확률 감소율
            use_target_net: Target Network 사용 여부
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)
        
        # Q-Network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Target Network (선택적)
        self.use_target_net = use_target_net
        if use_target_net:
            self.target_network = QNetwork(state_dim, action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_update_freq = 10  # 10 에피소드마다 업데이트
            self.update_counter = 0
    
    def get_action(self, state, valid_actions=None, training=True):
        """
        행동 선택 (Epsilon-greedy)
        
        Args:
            state: 현재 상태 (numpy array or list)
            valid_actions: 유효한 행동 리스트 (None이면 모두 가능)
            training: 학습 모드 여부
        
        Returns:
            선택된 행동 (int)
        """
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            # 랜덤 행동 (탐험)
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.action_dim - 1)
        
        # Q값 기반 행동 선택 (활용)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        
        # 유효한 행동만 고려
        if valid_actions:
            masked_q = np.full(self.action_dim, -np.inf)
            for action in valid_actions:
                masked_q[action] = q_values[action]
            return np.argmax(masked_q)
        
        return np.argmax(q_values)
    
    def train(self, batch):
        """
        배치 학습
        
        Args:
            batch: [(state, action, reward, next_state, done), ...]
        """
        if len(batch) == 0:
            return 0.0
        
        # 배치 데이터 변환
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # 현재 Q값 예측
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 목표 Q값 계산 (Bellman Equation)
        with torch.no_grad():
            if self.use_target_net:
                next_q_values = self.target_network(next_states).max(1)[0]
            else:
                next_q_values = self.q_network(next_states).max(1)[0]
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 손실 계산 및 역전파
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_epsilon(self):
        """Epsilon 감소"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """Target Network 업데이트"""
        if self.use_target_net:
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"모델 저장: {path}")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"모델 로드: {path}")
