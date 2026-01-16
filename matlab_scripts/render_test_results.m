function render_test_results(payload_path, save_path)
% RENDER_TEST_RESULTS 渲染测试结果可视化
%   从 .mat 文件加载测试数据并生成对比图
%
%   输入参数:
%       payload_path - .mat 文件路径，包含 images, masks, preds, dice, iou
%       save_path - 输出图像保存路径

data = load(payload_path);
images = data.images;
masks = data.masks;
preds = data.preds;
diceVals = data.dice;
iouVals = data.iou;

numSamples = size(images, 4);
fig = figure('Visible','off');
tiledlayout(fig, numSamples, 4, 'Padding','compact','TileSpacing','compact');

for idx = 1:numSamples
    img = images(:,:,:,idx);
    mask = masks(:,:,idx) > 0.5;
    pred = preds(:,:,idx) > 0.5;
    overlay = img;
    overlay(:,:,1) = max(overlay(:,:,1), mask);
    overlay(:,:,2) = max(overlay(:,:,2), pred);
    overlay(:,:,3) = max(overlay(:,:,3), mask & pred);
    
    nexttile; imshow(img, []); title(sprintf('样本 %d 原图', idx));
    nexttile; imshow(mask); title('真实Mask');
    nexttile; imshow(pred); title(sprintf('预测Mask\nDice %.3f / IoU %.3f', diceVals(idx), iouVals(idx)));
    nexttile; imshow(overlay); title('叠加对比');
end

exportgraphics(fig, save_path, 'Resolution', 300);
close(fig);

end

