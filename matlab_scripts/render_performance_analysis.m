function render_performance_analysis(group_means, group_stds, metric_names, save_path)
% RENDER_PERFORMANCE_ANALYSIS 渲染性能分析柱状图
%   绘制性能指标的柱状图（带误差棒）
%
%   输入参数:
%       group_means - 指标均值向量（列向量）
%       group_stds - 指标标准差向量（列向量）
%       metric_names - 指标名称字符串数组
%       save_path - 输出图像保存路径

% 确保数据是列向量
if size(group_means, 2) > size(group_means, 1)
    group_means = group_means';
end
if size(group_stds, 2) > size(group_stds, 1)
    group_stds = group_stds';
end

x_axis = 1:length(group_means);

% 【排版修复】创建高清画布：1200x800 像素
% 增加高度以容纳倾斜的 X 轴标签
fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1200, 900]);

% 【关键修复】调整坐标轴位置，给 X 轴标签预留更多空间（35% 而不是 25%）
% [left, bottom, width, height] - 使用归一化坐标 (0-1)
% bottom=0.35 给 X 轴标签留 35% 的高度，确保倾斜标签完全可见
ax = axes('Position', [0.12, 0.35, 0.80, 0.55]);

% === 【强制修复】转换标签数据类型（在绘图前准备） ===
% 使用独立的变量名，避免后续被覆盖
try
    % 尝试将各种格式转换为 cell array（MATLAB 最兼容的格式）
    if iscell(metric_names)
        x_tick_labels = metric_names;
    elseif isstring(metric_names)
        x_tick_labels = cellstr(metric_names);  % 转换为 cell array
    elseif ischar(metric_names)
        if size(metric_names, 1) > 1
            % 字符矩阵，转换为 cell array
            x_tick_labels = cellstr(metric_names);
        else
            % 单个字符串
            x_tick_labels = {metric_names};
        end
    else
        % 其他格式，尝试转换
        x_tick_labels = cellstr(string(metric_names));
    end
catch
    % 兜底：如果转换失败，直接使用硬编码标签
    warning('metric_names 转换失败，使用硬编码标签');
    x_tick_labels = {'Dice', 'IoU', 'Precision', 'Recall'};
end

% 确保标签数量匹配
if length(x_tick_labels) ~= length(group_means)
    warning('标签数量不匹配，使用硬编码标签');
    x_tick_labels = {'Dice', 'IoU', 'Precision', 'Recall'};
    x_tick_labels = x_tick_labels(1:min(length(x_tick_labels), length(group_means)));
end

% 输出调试信息
disp(['[MATLAB] X轴标签数量: ', num2str(length(x_tick_labels))]);
disp(['[MATLAB] X轴标签内容: ', strjoin(x_tick_labels, ', ')]);

% === 绘制柱状图（使用 categorical 确保标签正确显示） ===
% 【关键修复】使用 categorical 数组作为 X 轴数据，这样可以自动设置标签
use_categorical = false;  % 标记是否使用 categorical
try
    % 尝试使用 categorical（MATLAB R2013b+）
    x_categorical = categorical(x_tick_labels, x_tick_labels, 'Protected', true);
    b = bar(ax, x_categorical, group_means);
    use_categorical = true;
    disp('[MATLAB] 使用 categorical 数组绘制柱状图');
catch
    % 如果 categorical 不可用，使用传统方法
    b = bar(ax, x_axis, group_means);
    disp('[MATLAB] 使用传统方法绘制柱状图');
end

b.FaceColor = [0.30, 0.75, 0.93];  % 使用更鲜明的蓝色
b.EdgeColor = 'none';
b.FaceAlpha = 0.7;
hold(ax, 'on');

% 绘制误差棒
% 注意：如果使用 categorical，需要转换回数值位置
try
    if exist('x_categorical', 'var')
        % 使用 categorical 时，误差棒也需要使用 categorical
        er = errorbar(ax, x_categorical, group_means, group_stds);
    else
        er = errorbar(ax, x_axis, group_means, group_stds);
    end
catch
    er = errorbar(ax, x_axis, group_means, group_stds);
end
er.Color = [0.2, 0.2, 0.2];
er.LineStyle = 'none';
er.LineWidth = 1.5;
er.CapSize = 10;

% 美化图表
title(ax, 'Performance Metrics Analysis', 'FontSize', 16, 'FontWeight', 'bold');
ylabel(ax, 'Metric Value', 'FontSize', 14);
xlabel(ax, 'Metric Name', 'FontSize', 14);
grid(ax, 'on');
set(ax, 'GridAlpha', 0.15);
set(ax, 'LineWidth', 1.2);

% === 【强制修复】强制设置坐标轴标签（使用多种方法确保成功） ===
if use_categorical
    % 【使用 categorical 时】标签已经自动设置，只需要设置样式
    % 注意：不能使用数值设置 XTick，必须使用 categorical 值
    try
        xtickangle(ax, 45);  % 倾斜标签
        set(ax, 'FontSize', 12, 'FontName', 'Arial');
        disp('[MATLAB] categorical 模式下，标签已自动设置，仅设置样式');
    catch ME
        warning(['categorical 模式下设置样式失败: ', ME.message]);
    end
else
    % 【传统方法】使用数值设置 XTick 和标签
    xtick_positions = 1:length(group_means);
    set(ax, 'XTick', xtick_positions);
    
    % 【方法1】使用 set 函数设置 XTickLabel（传统方法）
    set(ax, 'XTickLabel', x_tick_labels);  % 使用 cell array
    
    % 【方法2】使用 xticklabels 函数（MATLAB R2016b+，推荐方法）
    try
        xticklabels(ax, x_tick_labels);
        disp('[MATLAB] 使用 xticklabels 函数设置标签成功');
    catch
        warning('xticklabels 函数不可用，使用传统方法');
    end
    
    % 【方法3】使用 xtickangle 倾斜标签
    xtickangle(ax, 45);  % 倾斜标签
    
    % 增加字体大小确保可见
    set(ax, 'FontSize', 12, 'FontName', 'Arial');
end

% 自动调整 y 轴范围
y_max = max(group_means + group_stds) * 1.1;
y_min = min(group_means - group_stds) * 0.9;
if y_min < 0
    y_min = 0;
end
ylim(ax, [y_min, y_max]);

% 添加数值标签（优化字体大小）
% 注意：使用不同的变量名，避免覆盖 x_tick_labels
xtips = b.XEndPoints;
ytips = b.YEndPoints;
value_labels = string(round(b.YData, 3));
text(ax, xtips, ytips, value_labels, 'HorizontalAlignment','center',...
    'VerticalAlignment','bottom', 'FontSize', 11, 'FontWeight','bold');

% 【关键修复】在所有绘图操作之后，再次强制设置 X 轴标签，确保不被覆盖
if use_categorical
    % 【使用 categorical 时】只需要设置样式，标签已经自动设置
    try
        xtickangle(ax, 45);  % 倾斜 45 度防止重叠
        set(ax, 'FontSize', 12, 'FontName', 'Arial');  % 确保字体可见
    catch
        % 如果失败，继续
    end
else
    % 【传统方法】再次设置标签
    set(ax, 'XTick', xtick_positions);
    set(ax, 'XTickLabel', x_tick_labels);  % 使用 cell array
    
    % 再次使用 xticklabels 函数（推荐方法）
    try
        xticklabels(ax, x_tick_labels);
    catch
        % 如果失败，继续使用传统方法
    end
    
    xtickangle(ax, 45);  % 倾斜 45 度防止重叠
    set(ax, 'FontSize', 12, 'FontName', 'Arial');  % 确保字体可见
end

% 验证标签是否设置成功
current_labels = get(ax, 'XTickLabel');
disp(['[MATLAB] 当前 X 轴标签: ', num2str(length(current_labels)), ' 个']);
if iscell(current_labels)
    disp(['[MATLAB] 标签内容: ', strjoin(current_labels, ', ')]);
end

% 【排版修复】确保图表布局正确
set(ax, 'Box', 'on');  % 显示坐标轴边框

% 【最后修复】在保存前最后一次设置 X 轴标签，防止 exportgraphics 重置
if use_categorical
    % 【使用 categorical 时】只需要设置样式
    try
        xtickangle(ax, 45);
        set(ax, 'FontSize', 12, 'FontName', 'Arial');
    catch
        % 如果失败，继续
    end
else
    % 【传统方法】设置标签
    set(ax, 'XTick', xtick_positions);
    set(ax, 'XTickLabel', x_tick_labels);
    
    % 使用 xticklabels 函数（推荐方法）
    try
        xticklabels(ax, x_tick_labels);
    catch
        % 如果失败，继续使用传统方法
    end
    
    xtickangle(ax, 45);
    set(ax, 'FontSize', 12, 'FontName', 'Arial');
end

% 【关键修复】确保 X 轴标签可见性
% 设置 X 轴标签的字体大小和颜色，确保可见
ax.XAxis.FontSize = 12;
ax.XAxis.FontWeight = 'bold';
ax.XAxis.Color = [0 0 0];  % 黑色

% 【额外修复】直接设置 X 轴刻度标签对象的属性
try
    % 获取 X 轴刻度标签对象并直接设置
    ax.XAxis.TickLabelRotation = 45;
    ax.XAxis.TickLabelInterpreter = 'none';  % 不使用 LaTeX 解释器
catch
    % 如果失败，继续使用传统方法
end

% 验证标签是否设置成功（保存前）
current_labels = get(ax, 'XTickLabel');
disp(['[MATLAB] 保存前 X 轴标签数量: ', num2str(length(current_labels))]);
if iscell(current_labels)
    disp(['[MATLAB] 保存前标签内容: ', strjoin(current_labels, ', ')]);
elseif ischar(current_labels)
    disp(['[MATLAB] 保存前标签内容: ', current_labels]);
end

% 【关键修复】优先使用 print 函数保存，它更好地保留坐标轴设置
% exportgraphics 可能会重置某些坐标轴设置，导致标签丢失
disp('正在导出性能分析图...');

% 方法1：优先使用 print（更可靠地保留坐标轴设置）
try
    % 确保路径是字符串格式
    if ischar(save_path) || isstring(save_path)
        save_path_str = char(save_path);
    else
        save_path_str = save_path;
    end
    
    % 使用 print 保存（300 DPI，PNG 格式）
    print(fig, save_path_str, '-dpng', '-r300');
    disp('[MATLAB] 使用 print 保存成功（保留坐标轴设置）');
catch ME
    % 如果 print 失败，尝试使用 exportgraphics
    warning(['print 失败: ', ME.message, '，尝试使用 exportgraphics']);
    try
        exportgraphics(fig, save_path, 'Resolution', 300, 'BackgroundColor', 'white');
        disp('[MATLAB] 使用 exportgraphics 保存成功');
    catch ME2
        error(['保存失败: ', ME2.message]);
    end
end

% 【最终验证】保存后再次检查（虽然已经关闭，但用于调试）
% disp(['[MATLAB] 图片已保存，X轴标签应显示: ', strjoin(x_tick_labels, ', ')]);

close(fig);

disp('性能分析图已成功生成');

end

