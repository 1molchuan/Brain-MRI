# -*- coding: utf-8 -*-
import utils.logging_setup
"""
预测线程模块
"""
from worker.common import *

class PredictThread(QThread):
    update_progress = pyqtSignal(int, str)
    prediction_finished = pyqtSignal(list, list, list)  # 添加原始图像路径参数
    
    def __init__(self, image_paths, model_path, threshold=0.5, save_results=True, output_dir=None):
        super().__init__()
        self.image_paths = image_paths
        self.model_path = model_path
        self.threshold = threshold
        self.save_results = save_results
        self.output_dir = output_dir
        if self.save_results and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        self.model_config = read_checkpoint_config(model_path) if model_path else None
        self.model_type = (self.model_config or {}).get("model_type", "improved_unet")
        self.swin_params = (self.model_config or {}).get("swin_params")
        self.dstrans_params = (self.model_config or {}).get("dstrans_params")
        self.model_threshold = (self.model_config or {}).get("best_threshold")
        if self.model_threshold is not None:
            self.threshold = float(self.model_threshold)
        self.use_tta = True
        context_cfg = (self.model_config or {}).get("context") or {}
        self.context_slices = int(context_cfg.get("slices", os.environ.get("SEG_CONTEXT_SLICES", "0")))
        self.context_gap = int(context_cfg.get("gap", os.environ.get("SEG_CONTEXT_GAP", "1")))
        self.required_modalities = (self.model_config or {}).get("extra_modalities") or []
        self.extra_modalities_dirs = parse_extra_modalities_spec(os.environ.get("SEG_EXTRA_MODALITIES"))
        if self.required_modalities:
            missing = [m for m in self.required_modalities if m not in self.extra_modalities_dirs]
            if missing:
                print(f"[提示] 模型期望额外模态: {missing}，当前未在 SEG_EXTRA_MODALITIES 中配置，将尝试仅使用可用模态。")
        skull_cfg = (self.model_config or {}).get("skull_stripping") or {}
        self.use_skull_stripper = skull_cfg.get("enabled", False)
        self.skull_stripper_path = skull_cfg.get("model_path")
        self.skull_stripper_threshold = skull_cfg.get("threshold", 0.5)
        if self.use_skull_stripper and not self.skull_stripper_path:
            self.use_skull_stripper = False
        # nnFormer 配置
        self.use_nnformer = False
        # 智能后处理配置
        self.smart_post_cfg = None
        try:
            if self.model_path:
                import json as _json
                model_dir = os.path.dirname(self.model_path)
                cfg_path = os.path.join(model_dir, "best_postprocessing_config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        self.smart_post_cfg = _json.load(f)
                    print(f"[SmartPost][Predict] 加载最佳后处理配置: {cfg_path} -> {self.smart_post_cfg}")
                else:
                    print("[SmartPost][Predict] 未找到 best_postprocessing_config.json，将使用默认后处理逻辑。")
        except Exception as e:
            print(f"[SmartPost][Predict] 加载最佳后处理配置失败，将使用默认后处理逻辑: {e}")
    
    def _predict_with_tta(self, model, image, use_tta=True):
        import torch.nn.functional as F
        if not use_tta:
            return torch.sigmoid(model(image))
        preds = []
        preds.append(torch.sigmoid(model(image)))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[3]))), dims=[3]))
        preds.append(torch.flip(torch.sigmoid(model(torch.flip(image, dims=[2]))), dims=[2]))
        preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(image, k=1, dims=[2, 3]))), k=-1, dims=[2, 3]))
        
        # 【关键修复】统一所有预测的空间尺寸
        if len(preds) > 0 and preds[0].dim() == 4:
            _, _, H, W = preds[0].shape
            target_size = (H, W)
            normalized_preds = []
            for pred in preds:
                if pred.dim() == 4:
                    _, _, h, w = pred.shape
                    if h != H or w != W:
                        # 插值到目标尺寸
                        pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=False)
                normalized_preds.append(pred)
            preds = normalized_preds
        
        return torch.stack(preds, dim=0).mean(dim=0)
    
    def _post_process(self, prob_tensor):
        # 先执行默认后处理
        processed = TrainThread.post_process_mask(
            prob_tensor.squeeze(0), 
            min_size=150, 
            use_morphology=True,
            keep_largest=False,  # 允许多发病灶同时存在
            fill_holes=True,     # 填充孔洞，去除假阴性空洞
            prob_map=prob_tensor.squeeze(0)
        )
        # 然后根据智能策略进一步调整
        if self.smart_post_cfg:
            try:
                from utils.smart_postprocessing import _apply_strategy  # type: ignore
                mtd = self.smart_post_cfg.get("method", "baseline")
                pms = self.smart_post_cfg.get("params", {})
                processed_np = processed.detach().cpu().numpy() if isinstance(processed, torch.Tensor) else np.asarray(processed, dtype=np.float32)
                processed_np = _apply_strategy(processed_np, mtd, pms)
                processed = torch.from_numpy(processed_np).float()
            except Exception as e:
                print(f"[SmartPost][Predict] 应用智能策略失败，回退默认后处理: {e}")
        if isinstance(processed, torch.Tensor):
            return processed.unsqueeze(0).unsqueeze(0)
        processed = torch.from_numpy(processed).float()
        return processed.unsqueeze(0).unsqueeze(0)
    

    def run(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.update_progress.emit(0, f"使用设备: {device}")

            
            # 数据转换
            transform = A.Compose([
                A.Resize(512, 512),  # 提升分辨率以保留更多病灶边缘细节
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
            # 创建数据集
            extra_modalities = build_extra_modalities_lists(self.image_paths, self.extra_modalities_dirs)
            dataset = MedicalImageDataset(
                self.image_paths,
                transform=transform,
                training=False,
                extra_modalities=extra_modalities,
                context_slices=self.context_slices,
                context_gap=self.context_gap
            )
            # 【Windows 多进程优化】为预测线程也启用多进程数据加载
            # Intel Core Ultra 9 285HX: 使用 8 个 worker 充分利用 P-Core
            import platform
            is_windows = platform.system() == 'Windows'
            cpu_count = os.cpu_count() or 1
            num_workers = 8 if is_windows else max(0, min(4, cpu_count - 1))
            dataloader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == 'cuda'),  # 【优化】加速数据传输
                persistent_workers=(num_workers > 0)  # 【关键】让子进程保持存活
            )
            if self.model_threshold is not None:
                self.update_progress.emit(8, f"使用模型自适应阈值: {self.threshold:.3f}")
            
            # 加载分割模型 - 使用兼容加载
            model = instantiate_model(self.model_type, device, self.swin_params, self.dstrans_params, None)
            success, msg = load_model_compatible(model, self.model_path, device, verbose=True)
            if not success:
                raise RuntimeError(f"模型加载失败: {msg}")
            model.eval()
            skull_stripper = None
            if self.use_skull_stripper:
                skull_stripper = SkullStripper(self.skull_stripper_path, device, self.skull_stripper_threshold)
                if not skull_stripper.is_available():
                    skull_stripper = None
                    self.update_progress.emit(6, "SkullStripper不可用，回退为单阶段推理")
            
            self.update_progress.emit(10, "模型加载完成，开始预测...")
            
            input_images = []
            output_masks = []
            input_numpy_images = []  # 存储原始图像数据
            
            with torch.no_grad():
                for i, batch_data in enumerate(dataloader):
                    # 处理数据
                    if isinstance(batch_data, tuple):
                        if len(batch_data) == 2:
                            image, mask = batch_data
                        else:
                            image = batch_data[0]
                    else:
                        image = batch_data
                    # 确保image是tensor
                    if not isinstance(image, torch.Tensor):
                        if isinstance(image, (list, tuple)) and len(image) > 0:
                            image = image[0]
                    image = image.to(device)
                    brain_mask = None
                    if skull_stripper and skull_stripper.is_available():
                        image, brain_mask = skull_stripper.strip(image)
                    
                    # 分割预测
                    prob = self._predict_with_tta(model, image, use_tta=self.use_tta)
                    if brain_mask is not None:
                        prob = prob * brain_mask
                    pred = (prob > self.threshold).float()
                    pred = self._post_process(pred)
                    
                    # 转换回图像格式
                    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                    image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                    prob_np = prob[0, 0].cpu().numpy()
                    pred_np = pred[0, 0].cpu().numpy()
                    pred_np = (pred_np * 255).astype(np.uint8)
                    
                    # 存储原始图像数据
                    input_numpy_images.append((image_np, pred_np, prob_np, ""))
                    
                    # 如果需要保存结果
                    if self.save_results and self.output_dir:
                        # 安全获取文件名
                        if i < len(self.image_paths):
                            base_name = os.path.splitext(os.path.basename(self.image_paths[i]))[0]
                        else:
                            base_name = f"image_{i}"
                        input_path = os.path.join(self.output_dir, f"{base_name}_input.png")
                        output_path = os.path.join(self.output_dir, f"{base_name}_mask.png")
                        cv2.imwrite(input_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(output_path, pred_np)
                        
                        input_images.append(input_path)
                        output_masks.append(output_path)
                    else:
                        # 如果不保存，使用临时文件名
                        input_images.append(f"image_{i}_input")
                        output_masks.append(f"image_{i}_mask")
                    
                    progress_msg = f"处理图像 {i+1}/{len(dataloader)}"
                    progress = 10 + int(90 * (i + 1) / len(dataloader))
                    self.update_progress.emit(progress, progress_msg)
            
            self.prediction_finished.emit(input_images, output_masks, input_numpy_images)
        
        except Exception as e:
            self.update_progress.emit(0, f"预测错误: {str(e)}")




