% 倒立摆控制：强化学习算法 奖励 (reward) 对比
clear; clc; close all;

% --- 1. 读取数据 ---
ppo_data = readtable('ppo_metrics.csv');
a2c_data = readtable('a2c_metrics.csv');
reinforce_data = readtable('reinforce_metrics.csv');

ppo_reward = ppo_data{:, 2};
a2c_reward = a2c_data{:, 2};
reinforce_reward = reinforce_data{:, 2};

episodes = 1:1500;

% --- 2. 计算滑动平均 ---
window_size = 50; 
ppo_smooth = movmean(ppo_reward, window_size);
a2c_smooth = movmean(a2c_reward, window_size);
reinforce_smooth = movmean(reinforce_reward, window_size);

% --- 3. 配置绘图参数 ---
figure('Position', [100, 100, 1000, 700], 'Color', 'w');
sgtitle('Learning Curves of Different RL Algorithms', 'FontSize', 16, 'FontWeight', 'bold');
fill_alpha = 0.3; % 阴影透明度

% 子图 1: PPO 
subplot(2, 2, 1);
hold on; grid on; box on;
plot(episodes, ppo_reward, 'Color', [0.8500, 0.3250, 0.0980, fill_alpha], 'LineWidth', 0.5); % 原始阴影
plot(episodes, ppo_smooth, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);             % 平滑主线
yline(500, 'k--', 'LQR Optimal (500)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
title('PPO (Proximal Policy Optimization)', 'FontSize', 12);
xlabel('Episodes'); ylabel('Total Reward');
xlim([0 1500]); ylim([0 520]);

% 子图 2: A2C 
subplot(2, 2, 2);
hold on; grid on; box on;
plot(episodes, a2c_reward, 'Color', [0.9290, 0.6940, 0.1250, fill_alpha], 'LineWidth', 0.5);
plot(episodes, a2c_smooth, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2);
yline(500, 'k--', 'LQR Optimal (500)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
title('A2C (with MC Returns)', 'FontSize', 12);
xlabel('Episodes'); ylabel('Total Reward');
xlim([0 1500]); ylim([0 520]);

% ==================== 子图 3: REINFORCE ====================
subplot(2, 2, 3);
hold on; grid on; box on;
plot(episodes, reinforce_reward, 'Color', [0, 0.4470, 0.7410, fill_alpha], 'LineWidth', 0.5);
plot(episodes, reinforce_smooth, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
yline(500, 'k--', 'LQR Optimal (500)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
title('REINFORCE', 'FontSize', 12);
xlabel('Episodes'); ylabel('Total Reward');
xlim([0 1500]); ylim([0 520]);

% 子图 4: 综合平滑对比 
subplot(2, 2, 4);
hold on; grid on; box on;
% 绘制三者平滑曲线
h1 = plot(episodes, ppo_smooth, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2);
h2 = plot(episodes, a2c_smooth, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2);
h3 = plot(episodes, reinforce_smooth, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2);
% 绘制并获取 LQR 基准线句柄，用于加入图例
h4 = yline(500, 'k--', 'LineWidth', 1.5); 
title('Smoothed Comparison', 'FontSize', 12);
xlabel('Episodes'); ylabel('Total Reward');
xlim([0 1500]); ylim([0 520]);
% 添加包含 LQR 的图例
legend([h1, h2, h3, h4], {'PPO', 'A2C', 'REINFORCE', 'LQR Baseline'}, 'Location', 'southeast', 'FontSize', 10);

% --- 4. 导出图片 ---
exportgraphics(gcf, 'Reward_Comparison.png', 'Resolution', 300);
disp('对比图已生成并保存为 Reward_Comparison.png');