import os
import math
import copy
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# =====================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 32          
batch_size = 256       
lr_rate = 3e-4         
epochs = 1000          
T = 1000               

# COSINE NOISE SCHEDULE IMPLEMENTATION
def get_cosine_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()

betas = get_cosine_schedule(T).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

# =====================================================================
# 2. HIGH-CAPACITY U-NET WITH TIME EMBEDDINGS, GROUP NORM & DUAL ATTENTION
# =====================================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.gn1 = nn.GroupNorm(32, out_ch)
        self.gn2 = nn.GroupNorm(32, out_ch)
        self.relu = nn.GELU()
        
    def forward(self, x, t):
        h = self.relu(self.gn1(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2] # Reshape to [B, C, 1, 1]
        h = h + time_emb
        return self.relu(self.gn2(self.conv2(h)) + h) # Residual connection

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1) # [B, H*W, C]
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        attn_out = attn_out + x_flat
        ff_out = self.ff(self.ln(attn_out)) + attn_out
        return ff_out.permute(0, 2, 1).view(B, C, H, W)

class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        self.inc = nn.Conv2d(3, 64, 3, padding=1)
        self.down1 = Block(64, 64, time_dim)
        self.down2 = Block(64, 128, time_dim)
        self.down3 = Block(128, 256, time_dim)
        
        self.attn2 = AttentionBlock(128)      
        self.attn3 = AttentionBlock(256)      
        
        self.up1 = Block(256 + 128, 128, time_dim)
        self.up2 = Block(128 + 64, 64, time_dim)
        self.up3 = Block(64, 64, time_dim)  
        self.outc = nn.Conv2d(64, 3, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        x1 = self.down1(self.inc(x), t)      # [32, 32, 64]
        x2 = self.down2(self.pool(x1), t)    # [16, 16, 128]
        
        x2 = self.attn2(x2)                  
        
        x3 = self.down3(self.pool(x2), t)    # [8, 8, 256]
        x3 = self.attn3(x3)                   
        
        out = self.upsample(x3)              # [16, 16, 256]
        out = torch.cat([out, x2], dim=1)    # [16, 16, 384]
        out = self.up1(out, t)               # [16, 16, 128]
        
        out = self.upsample(out)             # [32, 32, 128]
        out = torch.cat([out, x1], dim=1)    # [32, 32, 192]
        out = self.up2(out, t)               # [32, 32, 64]
        
        out = self.up3(out, t)               # [32, 32, 64] 
        return self.outc(out)

# =====================================================================
# 3. EMA (EXPONENTIAL MOVING AVERAGE) TRACKER
# =====================================================================
def update_ema(ema_model, model, decay=0.999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    alpha_hat = alpha_bar[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(alpha_hat) * x0 + torch.sqrt(1 - alpha_hat) * noise
    return xt, noise

# =====================================================================
# 4. ⚡ ULTRA-FAST DDIM SAMPLING LOOP (20x SPEEDUP FOR FID)
# =====================================================================
@torch.no_grad()
def generate_fid_samples_ddim(model, total_samples=10000, batch_size=250, ddim_steps=50, output_dir="fake_images"):
    model.eval()
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    times = torch.linspace(0, T - 1, ddim_steps, dtype=torch.long, device=device)
    times_prev = torch.cat([torch.tensor([-1], device=device), times[:-1]])
    
    img_counter = 0
    num_batches = math.ceil(total_samples / batch_size)
    print(f"\n⚡ Fast DDIM Sampler: Generating {total_samples} samples over {ddim_steps} steps...", flush=True)
    
    for b in range(num_batches):
        current_batch_size = min(batch_size, total_samples - img_counter)
        x = torch.randn(current_batch_size, 3, img_size, img_size, device=device)
        
        for idx in reversed(range(ddim_steps)):
            t = times[idx]
            t_prev = times_prev[idx]
            
            t_batch = torch.full((current_batch_size,), t, device=device, dtype=torch.long)
            noise_pred = model(x, t_batch)
            
            alpha_hat = alpha_bar[t]
            alpha_hat_prev = alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_hat) * noise_pred) / torch.sqrt(alpha_hat)
            pred_x0 = pred_x0.clamp(-1, 1)
            
            dir_xt = torch.sqrt(1 - alpha_hat_prev) * noise_pred
            x = torch.sqrt(alpha_hat_prev) * pred_x0 + dir_xt
            
        x = ((x + 1) / 2).clamp(0, 1)
        for i in range(current_batch_size):
            save_image(x[i].cpu(), f"{output_dir}/fake_{img_counter}.png")
            img_counter += 1
            
    print(f"✅ Fast images ready in '{output_dir}/'\n", flush=True)

# =====================================================================
# 5. MAIN PIPELINE EXECUTION
# =====================================================================
def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("real_images", exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    if len(os.listdir("real_images")) < 10000:
        real_counter = 0
        for imgs, _ in train_loader:
            for img in imgs:
                if real_counter >= 10000: break
                save_image((img + 1) / 2, f"real_images/real_{real_counter}.png")
                real_counter += 1
            if real_counter >= 10000: break

    # Initialize Core Network raw
    model = DDPM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr_rate, weight_decay=1e-4)

    # Deepcopy clean EMA model architecture
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    # 🚀 FIXED: Disabled torch.compile for stable cluster runtime execution
    # model = torch.compile(model)

    print("\n🚀 Starting Training Loop in Eager Mode...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, T, (x.shape[0],), device=device)
            
            xt, noise = forward_diffusion(x, t)
            noise_pred = model(xt, t)
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            update_ema(ema_model, model, decay=0.999)
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss / len(train_loader):.6f}", flush=True)

        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"checkpoints/ddpm_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
            }, checkpoint_path)
            
            generate_fid_samples_ddim(
                model=ema_model, 
                total_samples=10000, 
                batch_size=250, 
                ddim_steps=50, 
                output_dir=f"fake_images_epoch_{epoch+1}"
            )

if __name__ == "__main__":
    main()