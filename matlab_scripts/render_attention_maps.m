function render_attention_maps(payload_path, save_path)
% RENDER_ATTENTION_MAPS 渲染注意力热力图
%   从 .mat 文件加载注意力数据并生成热力图叠加
%
%   输入参数:
%       payload_path - .mat 文件路径，包含 images, masks, preds, att1/att2/att3/att4
%       save_path - 输出图像保存路径

disp('正在加载注意力数据...');
data = load(payload_path);
images = data.images;
masks = data.masks;
preds = data.preds;

% 检测可用的注意力层
att_layers = {};
if isfield(data, 'att1')
    att_layers{end+1} = 'att1';
end
if isfield(data, 'att2')
    att_layers{end+1} = 'att2';
end
if isfield(data, 'att3')
    att_layers{end+1} = 'att3';
end
if isfield(data, 'att4')
    att_layers{end+1} = 'att4';
end

if isempty(att_layers)
    error('未找到注意力层数据 (att1, att2, att3, att4)');
end

numSamples = min(size(images, 4), 4);
numLayers = length(att_layers);

% 设置画布：每行一个样本，每列一个注意力层 + 原图/GT/Pred
cols = 3 + numLayers;  % Input, GT, Pred, + 各注意力层
fig = figure('Visible','off', 'Color', 'w', 'Position', [100, 100, 200 * cols, 300 * numSamples]);
tl = tiledlayout(fig, numSamples, cols, 'Padding','compact', 'TileSpacing','none');

for idx = 1:numSamples
    % --- 数据预处理 ---
    raw_img = images(:,:,:,idx);
    
    % 转换为灰度图并增强对比度
    if size(raw_img, 3) == 3
        gray_img = rgb2gray(raw_img);
    else
        gray_img = raw_img;
    end
    base_img = imadjust(mat2gray(gray_img));
    base_img_rgb = cat(3, base_img, base_img, base_img);
    
    gt_mask = double(masks(:,:,idx));
    pred_mask = double(preds(:,:,idx));
    
    % --- 绘图 1: 原图 ---
    nexttile(tl);
    imshow(base_img, []);
    title(sprintf('Sample %d\nInput', idx), 'FontSize', 11, 'FontWeight', 'bold');
    
    % --- 绘图 2: Ground Truth ---
    nexttile(tl);
    imshow(base_img, []); hold on;
    green = cat(3, zeros(size(gt_mask)), ones(size(gt_mask)), zeros(size(gt_mask)));
    h = imshow(green);
    set(h, 'AlphaData', gt_mask * 0.3);
    title('Ground Truth', 'FontSize', 11);
    
    % --- 绘图 3: Prediction ---
    nexttile(tl);
    imshow(base_img, []); hold on;
    red = cat(3, ones(size(pred_mask)), zeros(size(pred_mask)), zeros(size(pred_mask)));
    h = imshow(red);
    set(h, 'AlphaData', pred_mask * 0.3);
    title('Prediction', 'FontSize', 11);
    
    % --- 绘图 4-N: 各注意力层热力图 ---
    for layer_idx = 1:numLayers
        layer_name = att_layers{layer_idx};
        att_map = data.(layer_name);
        
        % 获取当前样本的注意力图
        if size(att_map, 3) >= idx
            att_2d = double(att_map(:,:,idx));
        else
            att_2d = double(att_map(:,:,1));  % 回退到第一个
        end
        
        % 使用 imresize 将热力图放大到原图尺寸
        [H_orig, W_orig] = size(base_img);
        [H_att, W_att] = size(att_2d);
        if H_att ~= H_orig || W_att ~= W_orig
            att_2d = imresize(att_2d, [H_orig, W_orig], 'bilinear');
        end
        
        % 归一化到 [0, 1]
        att_norm = mat2gray(att_2d);
        
        % 使用 jet 配色方案
        att_colored = ind2rgb(uint8(att_norm * 255), jet(256));
        
        % 叠加在原图上 (Alpha=0.5)
        nexttile(tl);
        imshow(base_img, []); hold on;
        h_heat = imshow(att_colored);
        set(h_heat, 'AlphaData', att_norm * 0.5);  % 50% 透明度
        title(sprintf('Attention %s', layer_name), 'FontSize', 11);
    end
end

disp('正在导出注意力热力图...');
exportgraphics(fig, save_path, 'Resolution', 300, 'BackgroundColor','white');
close(fig);

end

