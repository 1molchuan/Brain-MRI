function render_prediction_grid(payload_path, save_path)
% RENDER_PREDICTION_GRID 渲染预测结果网格
%   从 .mat 文件加载数据并生成预测结果对比图
%
%   输入参数:
%       payload_path - .mat 文件路径，包含 images, masks, preds
%       save_path - 输出图像保存路径

disp('正在加载数据...');
data = load(payload_path);
images = data.images;
masks = data.masks;
preds = data.preds;

% 限制样本数
numSamples = min(size(images, 4), 4);

% 设置画布：加大分辨率，设置白色背景
fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 1400, 350 * numSamples]);
tl = tiledlayout(fig, numSamples, 4, 'Padding','compact', 'TileSpacing','none');

for idx = 1:numSamples
    % --- 数据预处理 ---
    raw_img = images(:,:,:,idx);
    
    % 1. 自动对比度增强 (解决灰蒙蒙的问题)
    if size(raw_img, 3) == 3
        gray_img = rgb2gray(raw_img);
    else
        gray_img = raw_img;
    end
    % 归一化并增强对比度
    base_img = imadjust(mat2gray(gray_img));
    % 转回 RGB 以便彩色叠加
    base_img_rgb = cat(3, base_img, base_img, base_img);
    
    gt_mask = double(masks(:,:,idx));
    pred_mask = double(preds(:,:,idx));
    
    % --- 绘图 1: 原图 ---
    nexttile(tl); 
    imshow(base_img, []); 
    title(sprintf('Sample %d Input', idx), 'FontSize', 12, 'FontWeight', 'bold');
    
    % --- 绘图 2: Ground Truth (绿色风格) ---
    nexttile(tl); 
    imshow(base_img, []); hold on;
    % 创建绿色透明蒙版
    green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
    h = imshow(green); 
    set(h, 'AlphaData', gt_mask * 0.3); % 30% 透明度
    title('Ground Truth (Green)', 'FontSize', 12);
    
    % --- 绘图 3: Prediction (红色风格) ---
    nexttile(tl); 
    imshow(base_img, []); hold on;
    % 创建红色透明蒙版
    red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
    h = imshow(red); 
    set(h, 'AlphaData', pred_mask * 0.3); 
    title('Prediction (Red)', 'FontSize', 12);
    
    % --- 绘图 4: 叠加对比 (医学标准) ---
    % 绿色=GT, 红色=Pred, 黄色=重叠(正确预测)
    nexttile(tl); 
    imshow(base_img, []); hold on;
    
    % 绘制 GT (绿色轮廓)
    [B_gt,L_gt] = bwboundaries(gt_mask > 0.5, 'noholes');
    for idx_k = 1:length(B_gt)
        boundary = B_gt{idx_k};
        plot(boundary(:,2), boundary(:,1), 'g', 'LineWidth', 1.5);
    end
    
    % 绘制 Pred (红色轮廓)
    [B_pred,L_pred] = bwboundaries(pred_mask > 0.5, 'noholes');
    for idx_k = 1:length(B_pred)
        boundary = B_pred{idx_k};
        plot(boundary(:,2), boundary(:,1), 'r--', 'LineWidth', 1.5);
    end
    
    % 添加图例说明
    title('Overlay (Green=GT, Red=Pred)', 'FontSize', 12);
end

disp('正在高保真导出...');
% 使用 exportgraphics 的 ContentType='vector' 可以获得更锐利的文字
exportgraphics(fig, save_path, 'Resolution', 300, 'BackgroundColor','white');
close(fig);

end

