import math
import random
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 0. Global Configuration
# ==========================================
# 显微镜成像通常受限于样品杆倾转范围 (Missing Wedge Problem)
# 典型的倾转范围是 -70度 到 +70度
MICROSCOPY_ANGLE_RANGE = (-70, 70) 
NUM_PROJECTIONS = 32 # 投影数量，模拟稀疏采样

if __name__ == "__main__":
    # Mac M1/M2/M3 使用 MPS 加速，其他使用 CUDA 或 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

# ==========================================
# 1. Physics Operators (PyTorch Native for Speed)
# ==========================================
class MicroscopyRadon(nn.Module):
    """
    一个基于 PyTorch grid_sample 实现的高速 Radon 变换算子。
    特点：
    1. 支持 GPU/MPS 加速（比 skimage 快很多）。
    2. 支持 Batch 处理。
    3. 可微分。
    """
    def __init__(self, image_size, angles, device):
        super().__init__()
        self.image_size = image_size
        self.angles = angles # degrees
        self.device = device
        self.grid = self._create_grid()
        
    def _create_grid(self):
        # 创建旋转矩阵
        theta = torch.tensor(self.angles, device=self.device, dtype=torch.float32)
        theta = torch.deg2rad(theta)
        N = len(theta)
        
        # 仿射变换矩阵 [N, 2, 3]
        # 旋转图像相当于坐标系的逆旋转
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # PyTorch 的 grid_sample 坐标系是 [-1, 1]
        # 旋转矩阵: [[cos, -sin, 0], [sin, cos, 0]]
        # 注意：这里我们旋转图像，然后沿着一个轴求和来模拟投影
        rotation_matrices = torch.zeros(N, 2, 3, device=self.device)
        rotation_matrices[:, 0, 0] = c
        rotation_matrices[:, 0, 1] = -s
        rotation_matrices[:, 1, 0] = s
        rotation_matrices[:, 1, 1] = c
        
        # 生成采样网格 [N, H, W, 2]
        grid = F.affine_grid(rotation_matrices, torch.Size((N, 1, self.image_size, self.image_size)), align_corners=True)
        return grid

    def forward(self, x):
        """
        x: [B, 1, H, W]
        Returns: [B, Num_Angles, Detector_Size] (Sinogram)
        """
        B, C, H, W = x.shape
        N = self.grid.shape[0] # number of angles
        
        # 扩展 x 以匹配角度数量: [B * N, 1, H, W]
        x_expanded = x.repeat_interleave(N, dim=0)
        
        # 扩展 grid 以匹配 Batch: [B * N, H, W, 2]
        grid_expanded = self.grid.repeat(B, 1, 1, 1)
        
        # 旋转图像: [B * N, 1, H, W]
        x_rotated = F.grid_sample(x_expanded, grid_expanded, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 沿着高度方向积分 (Radon Transform 的本质) -> [B * N, 1, W]
        projections = torch.sum(x_rotated, dim=2)
        
        # Reshape to [B, N, W] (Sinogram)
        sinogram = projections.view(B, N, W)
        return sinogram

class MicroscopyIRadon(nn.Module):
    """
    简单的反投影算子 (Backprojection)，用于数据一致性校正。
    注意：这只是 BP (Backprojection)，不是 FBP (Filtered BP)。
    在 Diffusion 的校正步骤中，我们通常只需要梯度的方向（即 BP）。
    """
    def __init__(self, image_size, angles, device):
        super().__init__()
        self.image_size = image_size
        self.angles = angles
        self.device = device
        self.grid = self._create_grid()

    def _create_grid(self):
        # 反投影需要逆旋转
        theta = torch.tensor(self.angles, device=self.device, dtype=torch.float32)
        theta = torch.deg2rad(theta)
        N = len(theta)
        
        c = torch.cos(theta)
        s = torch.sin(theta)
        
        # 逆旋转矩阵
        rotation_matrices = torch.zeros(N, 2, 3, device=self.device)
        rotation_matrices[:, 0, 0] = c
        rotation_matrices[:, 0, 1] = s # sign flip for inverse
        rotation_matrices[:, 1, 0] = -s # sign flip for inverse
        rotation_matrices[:, 1, 1] = c
        
        grid = F.affine_grid(rotation_matrices, torch.Size((N, 1, self.image_size, self.image_size)), align_corners=True)
        return grid

    def forward(self, sinogram):
        """
        sinogram: [B, N, W]
        Returns: [B, 1, H, W]
        """
        B, N, W = sinogram.shape
        
        # 将正弦图抹开成图片: [B * N, 1, H, W]
        # 每一列都复制充满整个 H 维度
        projections_expanded = sinogram.view(B*N, 1, 1, W).expand(B*N, 1, self.image_size, W)
        
        # 扩展 grid: [B * N, H, W, 2]
        grid_expanded = self.grid.repeat(B, 1, 1, 1)
        
        # 旋转回原始角度
        backprojected = F.grid_sample(projections_expanded, grid_expanded, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 叠加所有角度的贡献: [B, N, H, W] -> sum -> [B, 1, H, W]
        backprojected = backprojected.view(B, N, self.image_size, self.image_size)
        reconstruction = torch.sum(backprojected, dim=1, keepdim=True)
        
        # 简单的归一化，避免数值爆炸
        reconstruction = reconstruction / N
        return reconstruction

# ==========================================
# 2. Dataset (With Robust Normalization)
# ==========================================
class MoS2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=128, stride=64):
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.image_patches = []
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.root_dir):
            print(f"Path {self.root_dir} not found. Creating dummy data.")
            self.image_patches = [np.zeros((self.patch_size, self.patch_size), dtype=np.float32) for _ in range(10)]
            return

        for root, _, files in os.walk(self.root_dir):
            for file in sorted(files):
                if file.endswith('.h5'):
                    try:
                        with h5py.File(os.path.join(root, file), 'r') as f:
                            data = None
                            max_size = 0
                            def visit_func(name, obj):
                                nonlocal data, max_size
                                if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                                    if obj.size > max_size:
                                        max_size = obj.size
                                        data = obj[:]
                            f.visititems(visit_func)
                            
                            if data is not None:
                                img = data.astype(np.float32)
                                
                                # 1. Force Grayscale
                                if img.ndim == 3:
                                    if img.shape[2] == 3: img = np.mean(img, axis=2)
                                    else: img = img[:, :, 0]
                                
                                # 2. Robust Normalization (Percentile Clipping)
                                # 显微镜图像常有极亮噪点，使用 1%-99% 截断能显著提升对比度
                                p1 = np.percentile(img, 1)
                                p99 = np.percentile(img, 99)
                                img = np.clip(img, p1, p99)
                                
                                # 3. Min-Max Normalize to [0, 1]
                                if img.max() > img.min():
                                    img = (img - img.min()) / (img.max() - img.min())
                                
                                # 4. Patching
                                h, w = img.shape
                                if h >= self.patch_size and w >= self.patch_size:
                                    for y in range(0, h - self.patch_size + 1, self.stride):
                                        for x in range(0, w - self.patch_size + 1, self.stride):
                                            patch = img[y:y+self.patch_size, x:x+self.patch_size]
                                            self.image_patches.append(patch)
                                        
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        patch = self.image_patches[idx]
        img_pil = Image.fromarray((patch * 255).astype(np.uint8), mode='L')
        if self.transform:
            return self.transform(img_pil), 0 
        return transforms.ToTensor()(img_pil), 0

# ==========================================
# 3. Model (Deep UNet)
# ==========================================
def timestep_embedding(timesteps, dim=64, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(timesteps[:, None])], dim=-1)
    return emb

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(4, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_ch)
        self.time_emb = nn.Linear(time_dim, out_ch)

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_emb(t_emb)[..., None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.trans = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2, padding=1)
        self.block = ConvBlock(in_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.trans(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        h = torch.cat([x, skip], dim=1)
        h = self.block(h, t_emb)
        return h

class DeepUNet(nn.Module):
    def __init__(self, base_channels=32, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        
        self.stem = ConvBlock(1, base_channels, time_dim)
        self.down1 = ConvBlock(base_channels, base_channels * 2, time_dim, stride=2)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, time_dim, stride=2)
        self.down3 = ConvBlock(base_channels * 4, base_channels * 8, time_dim, stride=2)
        self.down4 = ConvBlock(base_channels * 8, base_channels * 8, time_dim, stride=2)
        
        self.mid = ConvBlock(base_channels * 8, base_channels * 8, time_dim)
        
        self.up4 = Up(base_channels * 16, base_channels * 4, time_dim)
        self.up3 = Up(base_channels * 8, base_channels * 2, time_dim)
        self.up2 = Up(base_channels * 4, base_channels, time_dim)
        self.up1 = Up(base_channels * 2, base_channels, time_dim)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        h0 = self.stem(x, t_emb)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(h1, t_emb)
        h3 = self.down3(h2, t_emb)
        h4 = self.down4(h3, t_emb)
        m = self.mid(h4, t_emb)
        out = self.out(self.up1(self.up2(self.up3(self.up4(m, h3, t_emb), h2, t_emb), h1, t_emb), h0, t_emb))
        return out

# ==========================================
# 4. Diffusion & Sampling Logic
# ==========================================
NUM_TIMESTEPS = 1000

def get_diffusion_params(device):
    betas = torch.linspace(1e-4, 0.02, NUM_TIMESTEPS, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas, "alphas_cumprod": alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance
    }

def q_sample(x_start, t, params, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    
    def extract(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    sqrt_cumprod = extract(params["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus = extract(params["sqrt_one_minus_alphas_cumprod"], t, x_start.shape)
    return sqrt_cumprod * x_start + sqrt_one_minus * noise

def p_sample(model, x, t, t_index, params):
    betas_t = params["betas"][t_index]
    sqrt_one_minus_t = params["sqrt_one_minus_alphas_cumprod"][t_index]
    sqrt_recip_alpha_t = params["sqrt_recip_alphas"][t_index]
    
    # Model predict noise
    pred_noise = model(x, t)
    model_mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_one_minus_t * pred_noise)
    
    if t_index == 0:
        return model_mean
    
    posterior_var_t = params["posterior_variance"][t_index]
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_var_t) * noise

# Data Consistency with PyTorch Radon
def data_consistency_step(x_est, sinogram_obs, radon_op, iradon_op, lam=0.1):
    # 1. Forward project current estimate
    # x_est: [B, 1, H, W] -> sinogram_pred: [B, N, W]
    sinogram_pred = radon_op(x_est)
    
    # 2. Calculate residual
    # 注意：真实观测值 y_obs 可能需要归一化或缩放以匹配算子的输出量级
    # 简单的做法是直接相减
    diff = sinogram_obs - sinogram_pred
    
    # 3. Backproject residual
    # diff: [B, N, W] -> correction: [B, 1, H, W]
    correction = iradon_op(diff)
    
    # 4. Update
    return x_est + lam * correction

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    root_dir = "./MoS2_Nanowire"
    image_size = 128
    batch_size = 32
    num_epochs = 100
    
    # Initialize Diffusion Params
    diff_params = get_diffusion_params(device)
    
    # Initialize Microscopy Operators (Angles: -70 to 70 for Missing Wedge)
    theta = np.linspace(MICROSCOPY_ANGLE_RANGE[0], MICROSCOPY_ANGLE_RANGE[1], NUM_PROJECTIONS, endpoint=True)
    radon_op = MicroscopyRadon(image_size, theta, device)
    iradon_op = MicroscopyIRadon(image_size, theta, device)
    
    # Load Data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    dataset = MoS2Dataset(root_dir, transform=transform, patch_size=image_size)
    
    if len(dataset) > 0:
        # 优化：Mac 上建议使用 num_workers=0 避免多进程开销和错误
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        print(f"Data loaded. Total patches: {len(dataset)}")
    else:
        print("No data found. Exiting.")
        train_loader = None
        
    # Model
    model = DeepUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Training Loop
    if train_loader:
        print("Starting training...")
        for epoch in range(1, num_epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
            epoch_loss = 0
            
            for imgs, _ in pbar:
                imgs = imgs.to(device)
                
                # Training Step
                optimizer.zero_grad()
                t = torch.randint(0, NUM_TIMESTEPS, (imgs.shape[0],), device=device).long()
                loss = F.mse_loss(model(q_sample(imgs, t, diff_params), t), torch.randn_like(imgs)) # Simplified loss cal for speed
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            # (Optional) Save checkpoint logic here
    
    # Demo Reconstruction
    if len(dataset) > 0:
        print("\nRunning reconstruction demo...")
        model.eval()
        
        # Take one sample
        gt_img, _ = dataset[0]
        gt_img = gt_img.unsqueeze(0).to(device) # [1, 1, 128, 128]
        
        # Simulate Microscopy Sinogram (Sparse + Limited Angle)
        with torch.no_grad():
            sinogram_obs = radon_op(gt_img)
        
        # SGM Reconstruction
        x = torch.randn_like(gt_img)
        for i in tqdm(reversed(range(NUM_TIMESTEPS)), desc="Sampling"):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            # 1. Reverse Diffusion
            with torch.no_grad():
                x = p_sample(model, x, t, i, diff_params)
            # 2. Data Consistency (Microscopy Physics)
            x = data_consistency_step(x, sinogram_obs, radon_op, iradon_op, lam=0.1)
            x = torch.clamp(x, 0, 1)
            
        # Visualize
        plt.figure(figsize=(12, 4))
        plt.subplot(131); plt.title("GT (Gray)"); plt.imshow(gt_img[0,0].cpu(), cmap='gray')
        plt.subplot(132); plt.title("Sinogram (-70 to 70)"); plt.imshow(sinogram_obs[0].cpu(), aspect='auto', cmap='gray')
        plt.subplot(133); plt.title("Recon"); plt.imshow(x[0,0].cpu(), cmap='gray')
        plt.show()
