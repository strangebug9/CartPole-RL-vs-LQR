% 倒立摆控制：强化学习算法 策略熵 (Entropy) 对比
clear; clc; close all;

% --- 1. 读取数据 ---
ppo_data = readtable('ppo_metrics.csv');
a2c_data = readtable('a2c_metrics.csv');
reinforce_data = readtable('reinforce_metrics.csv');

% 提取 Entropy 数组 
ppo_ent = ppo_data{:, 5};
a2c_ent = a2c_data{:, 5};
reinforce_ent = reinforce_data{:, 5};

episodes = 1:1500;

% --- 2. 计算滑动平均 (去除震荡) ---
window_size = 50; 
ppo_smooth = movmean(ppo_ent, window_size);
a2c_smooth = movmean(a2c_ent, window_size);
reinforce_smooth = movmean(reinforce_ent, window_size);

% --- 3. 绘制 Entropy 对比图 ---
figure('Position', [150, 150, 700, 450], 'Color', 'w');
hold on; grid on; box on;

% 只画平滑曲线
h1 = plot(episodes, ppo_smooth, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2.5);
h2 = plot(episodes, a2c_smooth, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2.5);
h3 = plot(episodes, reinforce_smooth, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2.5);

% 图表修饰
set(gca, 'FontSize', 12, 'FontName', 'Times New Roman');
title('Policy Entropy Decay (Exploration vs. Exploitation)', 'FontSize', 15);
xlabel('Episodes', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Entropy (Uncertainty)', 'FontSize', 12, 'FontWeight', 'bold');

xlim([0 1500]);
% 初始随机熵接近 0.69 (ln2)，设置 Y 轴合理范围
ylim([0.2 0.7]); 

legend([h1, h2, h3], {'PPO', 'A2C', 'REINFORCE'}, 'Location', 'southwest', 'FontSize', 11);

% 导出图片
exportgraphics(gcf, 'Entropy_Comparison.png', 'Resolution', 300);
disp('策略熵对比图已保存为 Entropy_Comparison.png');