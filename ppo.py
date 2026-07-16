import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# === 1. 超参数配置 ===
LR = 0.001
GAMMA = 0.99
EPS_CLIP = 0.2            
K_EPOCHS = 8              # 从 4 提升到 8，充分压榨每批数据的价值
ENTROPY_COEF = 0.01       
TOTAL_EPISODES = 1500
MAX_STEPS = 500
UPDATE_EPISODES = 5       # 每隔 5 个回合更新一次策略 (动态适应步数变化)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Actor-Critic 网络 ===
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# === 3. PPO 智能体 ===
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCriticNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        
        self.policy_old = ActorCriticNet(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, logprob = self.policy_old.act(state_tensor)
        
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob.item())
        return action

    def store_transition(self, r, done):
        self.rewards.append(r)
        self.dones.append(done)

    def update(self):
        if len(self.states) == 0:
            return 0.0, 0.0

        # --- 1. 计算折扣回报 (Rewards-to-go) ---
        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)
            
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-7)

        old_states = torch.FloatTensor(np.array(self.states)).to(device)
        old_actions = torch.LongTensor(self.actions).to(device)
        old_logprobs = torch.FloatTensor(self.logprobs).to(device)

        total_loss_record = 0.0
        entropy_record = 0.0

        # --- 2. PPO 核心: 多 Epoch 迭代优化 ---
        for _ in range(K_EPOCHS):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards_to_go - state_values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(state_values, rewards_to_go)
            
            loss = actor_loss + 0.5 * critic_loss - ENTROPY_COEF * dist_entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss_record += loss.item()
            entropy_record += dist_entropy.mean().item()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()

        return total_loss_record / K_EPOCHS, entropy_record / K_EPOCHS

# === 4. 主训练流程 ===
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(state_dim, action_dim)
    
    history_data = []
    current_loss = 0.0
    current_entropy = 0.69

    print("===== 开始 PPO 训练 =====")
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(MAX_STEPS):
            action = agent.choose_action(state)
            s_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(reward, done)
            
            ep_reward += reward
            ep_steps += 1
            state = s_next
            
            if done:
                break

        if (episode + 1) % UPDATE_EPISODES == 0:
            current_loss, current_entropy = agent.update()

        history_data.append([episode + 1, ep_reward, ep_steps, current_loss, current_entropy])
        
        if (episode + 1) % 50 == 0:
            avg_r = np.mean([x[1] for x in history_data[-50:]])
            avg_ent = np.mean([x[4] for x in history_data[-50:]])
            print(f"Episode: {episode+1:4d} | Avg Reward: {avg_r:6.2f} | Avg Entropy: {avg_ent:.3f} | Loss: {current_loss:.2f}")

        # 设置严格的收敛提前终止条件
        # if len(history_data) >= 50 and np.mean([x[1] for x in history_data[-50:]]) > 495:
        #     print(f"\n训练已完美收敛，在第 {episode+1} 回合提前终止。")
        #     break

    env.close()

    # === 5. 导出训练数据 ===
    history_data = np.array(history_data)
    header = "Episode, Reward, Steps, Loss, Entropy"
    np.savetxt('ppo_metrics.csv', history_data, delimiter=',', header=header, comments='', fmt='%.4f')
    
    print("数据已导出至 'ppo_metrics.csv'")