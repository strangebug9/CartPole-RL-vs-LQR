import gymnasium as gym
import numpy as np
import scipy.linalg

# ===================== 1. Gym 环境物理参数提取 =====================
# 根据 CartPole-v1 的底层物理引擎参数进行严格映射
g = 9.8           # 重力加速度
Mc = 1.0          # 小车质量 (masscart)
Mp = 0.1          # 摆杆质量 (masspole)
Mt = Mc + Mp      # 总质量
l = 0.5           # 摆杆半长 (length)

# ===================== 2. 构建状态空间矩阵 A 和 B =====================
# 依据工作点 theta=0 处的线性化推导结果
A = np.array([
    [0, 1, 0, 0],
    [0, 0, (Mp * g) / Mc, 0],
    [0, 0, 0, 1],
    [0, 0, (Mt * g) / (Mc * l), 0]
])

B = np.array([
    [0],
    [1 / Mc],
    [0],
    [-1 / (Mc * l)] 
])

# ===================== 3. 设计 LQR 权重矩阵 Q 和 R =====================
# Q 矩阵对角线元素对应惩罚 [x, x_dot, theta, theta_dot]
Q = np.diag([1.0, 1.0, 100.0, 10.0])

# R 矩阵惩罚控制力输入，设为极小值表示“推力非常廉价，随意使用”
R = np.array([[0.01]])

# 求解连续代数黎卡提方程 (CARE) 得到 P 矩阵
P = scipy.linalg.solve_continuous_are(A, B, Q, R)

# 计算最优线性状态反馈增益矩阵 K
K = np.linalg.inv(R).dot(B.T).dot(P)
print(f"LQR 求解完成！最优反馈增益矩阵 K = {K.flatten()}")

# ===================== 4. 主测试流程 =====================
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    TOTAL_TEST_EPISODES = 1500
    MAX_STEPS = 500
    history_data = []

    print("===== 开始 LQR 控制测试 =====")
    for episode in range(TOTAL_TEST_EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0
        
        for step in range(MAX_STEPS):
            # 将当前状态转为列向量
            x_vec = np.array(state).reshape(-1, 1)
            
            # 计算连续最优控制力 U = -KX
            u = -K.dot(x_vec)[0, 0]
            
            # 连续力映射为 Gym 离散动作 (Bang-Bang Control)
            action = 1 if u > 0 else 0
            
            s_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
                
            ep_reward += reward
            ep_steps += 1
            state = s_next
            
            if done:
                break
                
        # 记录硬性指标：回合数, 总奖励, 存活步数, Loss(无), Entropy(无)
        # 为了与 RL 的数据维度严格对齐，Loss 和 Entropy 填补为 0
        history_data.append([episode + 1, ep_reward, ep_steps, 0.0, 0.0])
        
        # 日志打印频率改为与 RL 相同的每 50 局一次
        if (episode + 1) % 50 == 0:
            avg_r = np.mean([x[1] for x in history_data[-50:]])
            print(f"Episode: {episode+1:4d} | Avg Reward: {avg_r:6.2f} | (LQR 基于数学解析解，无 Loss/Entropy)")

    env.close()

    # ===================== 5. 导出数据 =====================
    history_data = np.array(history_data)
    header = "Episode, Reward, Steps, Loss, Entropy"
    np.savetxt('lqr_metrics.csv', history_data, delimiter=',', header=header, comments='', fmt='%.4f')
    
    print("\nLQR 数据已成功导出至 'lqr_metrics.csv'")