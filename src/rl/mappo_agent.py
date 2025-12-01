"""
Worm Robot Simulation - MAPPO Agent
Multi-Agent Proximal Policy Optimization 구현
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorNetwork(nn.Module):
    """Actor Network (정책 네트워크)"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class CriticNetwork(nn.Module):
    """Critic Network (가치 네트워크)"""
    
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO)
    
    특징:
    - Actor-Critic 구조
    - Clipped Surrogate Objective
    - Generalized Advantage Estimation (GAE)
    - Multi-Agent 지원
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        num_agents,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device="cpu"
    ):
        """
        Args:
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원 (4: 전진, 시계, 반시계, STAY)
            num_agents: 에이전트(로봇) 수
            learning_rate: 학습률
            gamma: 할인율
            gae_lambda: GAE lambda 파라미터
            clip_epsilon: PPO clipping 범위
            entropy_coef: 엔트로피 보너스 계수
            value_loss_coef: Value loss 가중치
            max_grad_norm: Gradient clipping 최대값
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        
        # Actor & Critic Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 경험 버퍼
        self.reset_buffer()
    
    def reset_buffer(self):
        """에피소드 버퍼 초기화"""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def get_action(self, state, training=True):
        """
        행동 선택 (확률적 정책)
        
        Args:
            state: 현재 상태 (numpy array or list)
            training: 학습 모드 여부
        
        Returns:
            action: 선택된 행동 (int)
            log_prob: 로그 확률 (training=True일 때)
            value: 상태 가치 (training=True일 때)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
        
        # 행동 샘플링
        dist = Categorical(action_probs)
        action = dist.sample()
        
        if training:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()
        else:
            # 평가 모드: 가장 높은 확률의 행동 선택
            return action_probs.argmax().item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """경험 저장"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def compute_gae(self, next_value=0.0):
        """
        Generalized Advantage Estimation (GAE) 계산
        
        Args:
            next_value: 다음 상태의 가치 (에피소드 종료 시 0)
        
        Returns:
            advantages: GAE 어드밴티지
            returns: 리턴 값
        """
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        dones = np.array(self.buffer['dones'])
        
        advantages = []
        gae = 0
        
        # 역순으로 계산
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_val = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def train(self, num_epochs=4, batch_size=64):
        """
        PPO 학습
        
        Args:
            num_epochs: 학습 epoch 수
            batch_size: 미니배치 크기
        
        Returns:
            actor_loss: Actor 손실
            critic_loss: Critic 손실
            entropy: 엔트로피
        """
        if len(self.buffer['states']) == 0:
            return 0.0, 0.0, 0.0
        
        # GAE 계산
        advantages, returns = self.compute_gae()
        
        # Tensor 변환
        states = torch.FloatTensor(self.buffer['states']).to(self.device)
        actions = torch.LongTensor(self.buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 학습
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        dataset_size = len(states)
        
        for epoch in range(num_epochs):
            # 미니배치 생성
            indices = np.random.permutation(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 현재 정책으로 재평가
                action_probs = self.actor(batch_states)
                values = self.critic(batch_states).squeeze()
                
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO Clipped Surrogate Objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (MSE)
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # Total Loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # Backward
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # 버퍼 초기화
        self.reset_buffer()
        
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0.0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0.0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0.0
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def save(self, path):
        """모델 저장"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)
        print(f"모델 저장: {path}")
    
    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"모델 로드: {path}")

