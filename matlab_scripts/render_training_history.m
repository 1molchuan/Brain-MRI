function render_training_history(payload_path, save_path)
% RENDER_TRAINING_HISTORY 渲染训练历史曲线
%   从 .mat 文件加载训练历史数据并生成曲线图
%
%   输入参数:
%       payload_path - .mat 文件路径，包含 epochs, train_loss, val_loss, val_dice
%       save_path - 输出图像保存路径

data = load(payload_path);
epochs = data.epochs;
trainLoss = data.train_loss;
valLoss = data.val_loss;
valDice = data.val_dice;

fig = figure('Visible','off');
tiledlayout(fig,1,2,'Padding','compact','TileSpacing','compact');

nexttile;
plot(epochs, trainLoss, '-ob', 'LineWidth', 2); hold on;
plot(epochs, valLoss, '-or', 'LineWidth', 2);
title('训练/验证损失'); xlabel('轮次'); ylabel('Loss');
legend('训练','验证','Location','best'); grid on;

nexttile;
plot(epochs, valDice, '-og', 'LineWidth', 2);
title('验证Dice'); xlabel('轮次'); ylabel('Dice'); ylim([0 1]); grid on;

exportgraphics(fig, save_path, 'Resolution', 300);
close(fig);

end

