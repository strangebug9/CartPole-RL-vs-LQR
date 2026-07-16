import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# === 1. 超参数配置 ===
LR = 0.001
GAMMA = 0.99
ENTROPY_COEF = 0.01       # 策略熵系数 
TOTAL_EPISODES = 1500
MAX_STEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Actor-Critic 网络 ===
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        # Actor 独立
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # Critic 独立
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        return action_probs, state_value

# === 3. A2C 智能体 ===
class A2CAgent:
    def __init__(self, state_dim, action_dim):
        self.ac_net = ActorCriticNet(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=LR)
        
        # --- 轨迹缓存 ---
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def choose_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, _ = self.ac_net(state_tensor)
        
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def store_transition(self, s, a, r, s_next, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_next)
        self.dones.append(done)

    def update(self):
        if len(self.states) == 0:
            return 0.0, 0.0

        # 转换为张量进行批量运算
        s = torch.FloatTensor(np.array(self.states)).to(device)
        a = torch.LongTensor(self.actions).view(-1, 1).to(device)
        r = torch.FloatTensor(self.rewards).view(-1, 1).to(device)
        s_next = torch.FloatTensor(np.array(self.next_states)).to(device)
        
        # 注意：这里的 dones 只有在 terminated 时才是 1
        dones = torch.FloatTensor(self.dones).view(-1, 1).to(device)

        # 获取当前与下一状态的网络输出
        probs, values = self.ac_net(s)
        _, next_values = self.ac_net(s_next)

        # --- 核心计算区域 ---
        
        # 1. 计算 TD Target: r + gamma * V(s')
        # 如果是 truncated 结束的，dones为0，完美保留 next_values
        td_target = r + GAMMA * next_values * (1 - dones)
        
        # 2. 计算优势函数 (Advantage): TD Target - V(s)
        advantage = td_target - values
        
        # 优势函数标准化 (消除 A2C 剧烈震荡的法宝)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # 3. Critic 损失: 均方误差 MSE(V(s), TD Target)
        critic_loss = nn.functional.mse_loss(values, td_target.detach())

        # 4. Actor 损失: -log_prob * Advantage
        m = Categorical(probs)
        log_probs = m.log_prob(a.squeeze())
        actor_loss = -(log_probs * advantage.detach().squeeze()).mean()

        # 5. 策略熵 (鼓励探索)
        entropy = m.entropy().mean()

        # 6. 综合损失 = Actor Loss + Critic Loss - 探索奖励
        total_loss = actor_loss + critic_loss - ENTROPY_COEF * entropy

        # --- 反向传播与网络更新 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 提取指标
        ep_loss = total_loss.item()
        ep_entropy = entropy.item()

        # 清空轨迹缓存
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

        return ep_loss, ep_entropy

# === 4. 主训练流程 ===
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = A2CAgent(state_dim, action_dim)
    
    # 记录矩阵格式: [回合, 奖励, 步数, 损失, 策略熵]
    history_data = []

    print("===== 开始 A2C 训练 =====")
    for episode in range(TOTAL_EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(MAX_STEPS):
            action = agent.choose_action(state)
            s_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            

            # 只有杆子倒下才算真正失败，超时达到 500 步不算失败
            agent.store_transition(state, action, reward, s_next, terminated)
            
            ep_reward += reward
            ep_steps += 1
            state = s_next
            
            if done:
                break

        # 回合更新策略 (符合 REINFORCE 对比基准)
        loss, entropy = agent.update()
        history_data.append([episode + 1, ep_reward, ep_steps, loss, entropy])
        
        # 打印训练进度 (每 50 回合)
        if (episode + 1) % 50 == 0:
            avg_r = np.mean([x[1] for x in history_data[-50:]])
            avg_ent = np.mean([x[4] for x in history_data[-50:]])
            print(f"Episode: {episode+1:4d} | Avg Reward: {avg_r:6.2f} | Avg Entropy: {avg_ent:.3f} | Loss: {loss:.2f}")

        # 提前终止条件
        # if len(history_data) >= 50 and np.mean([x[1] for x in history_data[-50:]]) > 495:
        #     print(f"\n训练已完美收敛，在第 {episode+1} 回合提前终止。")
        #     break

    env.close()

    # === 5. 导出训练数据 ===
    history_data = np.array(history_data)
    header = "Episode, Reward, Steps, Loss, Entropy"
    np.savetxt('a2c_metrics.csv', history_data, delimiter=',', header=header, comments='', fmt='%.4f')
    
    print("\n数据已导出至 'a2c_metrics.csv'")