import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# === 1. 超参数配置 ===
LR = 0.001
GAMMA = 0.99
TOTAL_EPISODES = 1500  
MAX_STEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. 策略网络 (Policy Network) ===
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.out(x), dim=-1)

# === 3. REINFORCE 智能体 ===
class REINFORCEAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = PolicyNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        
        # --- 轨迹缓存 ---
        self.log_probs = []
        self.rewards = []
        self.entropies = []  # 策略熵 (衡量探索程度)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.policy_net(state)
        
        # 使用多项式分布进行随机采样 (探索)
        m = Categorical(probs)
        action = m.sample()
        
        self.log_probs.append(m.log_prob(action))
        self.entropies.append(m.entropy())  
        
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update(self):
        # 如果缓存为空则跳过
        if len(self.rewards) == 0:
            return 0.0, 0.0

        # --- 1. 计算折扣回报 (Rewards-to-go / G_t) ---
        G = np.zeros_like(self.rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * GAMMA + self.rewards[t]
            G[t] = running_add

        # 回报标准化 (极大地降低 REINFORCE 的方差，加速收敛)
        G = torch.FloatTensor(G).to(device)
        G = (G - G.mean()) / (G.std() + 1e-8)

        # --- 2. 计算策略损失 (Policy Loss) ---
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, G):
            # Loss = -log(pi(a|s)) * G_t
            # 负号是因为 PyTorch 默认是最小化，我们要最大化期望回报
            policy_loss.append(-log_prob * G_t)
        
        policy_loss = torch.cat(policy_loss).sum()

        # --- 3. 反向传播与网络更新 ---
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # --- 4. 提取本局平均指标 ---
        ep_loss = policy_loss.item()
        ep_entropy = torch.stack(self.entropies).mean().item()

        # --- 5. 清空轨迹缓存 ---
        self.log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
        
        return ep_loss, ep_entropy

# === 4. 主训练流程 ===
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(state_dim, action_dim)
    
    # 记录矩阵格式: [回合, 奖励, 步数, 损失, 策略熵]
    history_data = []

    print("===== 开始 REINFORCE 训练 =====")
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(MAX_STEPS):
            action = agent.choose_action(state)
            s_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 使用环境原本的 +1 奖励
            agent.store_reward(reward)
            
            ep_reward += reward
            ep_steps += 1
            state = s_next
            
            if done:
                break

        # 回合结束，进行策略更新 (REINFORCE 是基于回合 Monte Carlo 的算法，必须回合末更新)
        loss, entropy = agent.update()
        history_data.append([episode + 1, ep_reward, ep_steps, loss, entropy])
        
        # 打印训练进度 (每 50 回合)
        if (episode + 1) % 50 == 0:
            avg_r = np.mean([x[1] for x in history_data[-50:]])
            avg_ent = np.mean([x[4] for x in history_data[-50:]])
            print(f"Episode: {episode+1:4d} | Avg Reward: {avg_r:6.2f} | Avg Entropy: {avg_ent:.3f} | Loss: {loss:.2f}")

        # 收敛提前终止条件
        # if len(history_data) >= 50 and np.mean([x[1] for x in history_data[-50:]]) > 495:
        #     print(f"\n训练已完美收敛，在第 {episode+1} 回合提前终止。")
        #     break

    env.close()

    # === 5. 导出训练数据 ===
    history_data = np.array(history_data)
    header = "Episode, Reward, Steps, Loss, Entropy"
    np.savetxt('reinforce_metrics.csv', history_data, delimiter=',', header=header, comments='', fmt='%.4f')
    
    print("\n训练结束。数据已导出至 'reinforce_metrics.csv'")