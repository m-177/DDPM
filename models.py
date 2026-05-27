import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------
# 更好的时间嵌入（通用）
# -----------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.proj1 = nn.Linear(dim, dim * 4)
        self.proj2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        emb = self.proj1(emb)
        emb = self.act(emb)
        emb = self.proj2(emb)

        return emb


# -----------------------------
# 1D自注意力模块
# -----------------------------
class SelfAttention1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        B, C, L = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.reshape(B, C, L).permute(0, 2, 1)
        k = k.reshape(B, C, L)
        v = v.reshape(B, C, L).permute(0, 2, 1)

        # Attention
        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1).reshape(B, C, L)

        return x + self.proj(out)


# -----------------------------
# 2D自注意力模块（保留原版）
# -----------------------------
class SelfAttention2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return x + self.proj(out)


# -----------------------------
# 1D改进的ResBlock
# -----------------------------
class ImprovedResBlock1D(nn.Module):
    def __init__(self, ch_in, ch_out, temb_ch=512, dropout=0.1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # 第一个卷积块（1D）
        self.norm1 = nn.GroupNorm(32, ch_in)
        self.conv1 = nn.Conv1d(ch_in, ch_out, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, ch_out)

        # 第二个卷积块
        self.norm2 = nn.GroupNorm(32, ch_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(ch_out, ch_out, 3, padding=1)

        # 注意力（可选）
        if use_attention:
            self.attn = SelfAttention1D(ch_out)
        else:
            self.attn = nn.Identity()

        # 跳跃连接
        if ch_in != ch_out:
            self.shortcut = nn.Conv1d(ch_in, ch_out, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # 加入时间嵌入
        temb_out = self.temb_proj(self.act(temb))
        h = h + temb_out[:, :, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        # 注意力
        h = self.attn(h)

        return h + self.shortcut(x)


# -----------------------------
# 2D改进的ResBlock（保留原版）
# -----------------------------
class ImprovedResBlock2D(nn.Module):
    def __init__(self, ch_in, ch_out, temb_ch=512, dropout=0.1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.norm1 = nn.GroupNorm(32, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, ch_out)

        self.norm2 = nn.GroupNorm(32, ch_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)

        if use_attention:
            self.attn = SelfAttention2D(ch_out)
        else:
            self.attn = nn.Identity()

        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        temb_out = self.temb_proj(self.act(temb))
        h = h + temb_out[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.attn(h)

        return h + self.shortcut(x)


# -----------------------------
# 1D改进的UNet
# -----------------------------
class ImprovedDenoiseUNet1D(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            model_channels=64,
            channel_multipliers=[1, 2, 4, 4],
            num_res_blocks=2,
            dropout=0.1,
            use_attention_at_res=[False, False, True, True],
            device="cuda",
            condition_channels=None
    ):
        super().__init__()
        self.device = device
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        temb_ch = model_channels * 4
        self.time_embed = TimeEmbedding(temb_ch)

        if condition_channels is not None:
            in_channels = in_channels + condition_channels

        self.conv_in = nn.Conv1d(in_channels, model_channels, 3, padding=1)

        # Encoder (1D)
        self.encoder_blocks = nn.ModuleList()
        encoder_channels = []
        ch = model_channels

        for level, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ImprovedResBlock1D(
                        ch, out_ch, temb_ch, dropout,
                        use_attention=use_attention_at_res[level]
                    )
                )
                ch = out_ch
                encoder_channels.append(ch)

            if level != len(channel_multipliers) - 1:
                self.encoder_blocks.append(nn.Conv1d(ch, ch, 3, stride=2, padding=1))
                encoder_channels.append(ch)

        # Middle
        self.middle_blocks = nn.ModuleList([
            ImprovedResBlock1D(ch, ch, temb_ch, dropout, use_attention=True),
            ImprovedResBlock1D(ch, ch, temb_ch, dropout, use_attention=False)
        ])

        # Decoder (1D)
        self.decoder_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(
                    ImprovedResBlock1D(
                        ch + skip_ch, out_ch, temb_ch, dropout,
                        use_attention=use_attention_at_res[level] and i == 0
                    )
                )
                ch = out_ch

            if level != 0:
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv1d(ch, ch, 3, padding=1)
                    )
                )

        # 输出层
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv1d(ch, out_channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x, t, cond=None):
        temb = self.time_embed(t)

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.conv_in(x)

        skips = []
        for block in self.encoder_blocks:
            if isinstance(block, ImprovedResBlock1D):
                h = block(h, temb)
                skips.append(h)
            else:
                h = block(h)
                skips.append(h)

        for block in self.middle_blocks:
            h = block(h, temb)

        for block in self.decoder_blocks:
            if isinstance(block, ImprovedResBlock1D):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb)
            else:
                h = block(h)

        h = self.act(self.norm_out(h))
        pred_noise = self.conv_out(h)

        return pred_noise


# -----------------------------
# 2D改进的UNet（保留原版）
# -----------------------------
class ImprovedDenoiseUNet2D(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            model_channels=64,
            channel_multipliers=[1, 2, 4, 4],
            num_res_blocks=2,
            dropout=0.1,
            use_attention_at_res=[False, False, True, True],
            device="cuda",
            condition_channels=None
    ):
        super().__init__()
        self.device = device
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        temb_ch = model_channels * 4
        self.time_embed = TimeEmbedding(temb_ch)

        if condition_channels is not None:
            in_channels = in_channels + condition_channels

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        encoder_channels = []
        ch = model_channels

        for level, mult in enumerate(channel_multipliers):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ImprovedResBlock2D(
                        ch, out_ch, temb_ch, dropout,
                        use_attention=use_attention_at_res[level]
                    )
                )
                ch = out_ch
                encoder_channels.append(ch)

            if level != len(channel_multipliers) - 1:
                self.encoder_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                encoder_channels.append(ch)

        self.middle_blocks = nn.ModuleList([
            ImprovedResBlock2D(ch, ch, temb_ch, dropout, use_attention=True),
            ImprovedResBlock2D(ch, ch, temb_ch, dropout, use_attention=False)
        ])

        self.decoder_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                self.decoder_blocks.append(
                    ImprovedResBlock2D(
                        ch + skip_ch, out_ch, temb_ch, dropout,
                        use_attention=use_attention_at_res[level] and i == 0
                    )
                )
                ch = out_ch

            if level != 0:
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(ch, ch, 3, padding=1)
                    )
                )

        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x, t, cond=None):
        temb = self.time_embed(t)

        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        h = self.conv_in(x)

        skips = []
        for block in self.encoder_blocks:
            if isinstance(block, ImprovedResBlock2D):
                h = block(h, temb)
                skips.append(h)
            else:
                h = block(h)
                skips.append(h)

        for block in self.middle_blocks:
            h = block(h, temb)

        for block in self.decoder_blocks:
            if isinstance(block, ImprovedResBlock2D):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb)
            else:
                h = block(h)

        h = self.act(self.norm_out(h))
        pred_noise = self.conv_out(h)

        return pred_noise


# -----------------------------
# 统一的DenoiseUNet（自动选择维度）
# -----------------------------
class DenoiseUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ch=64, droprate=0.1,
                 groups=8, device="cuda", residual=False, data_dim='1d'):
        """
        Args:
            data_dim: '1d' 或 '2d'，选择模型维度
        """
        super().__init__()
        self.device = device
        self.residual = residual
        self.data_dim = data_dim

        if data_dim == '1d':
            self.unet = ImprovedDenoiseUNet1D(
                in_channels=in_channels,
                out_channels=out_channels,
                model_channels=ch,
                channel_multipliers=[1, 2, 4],
                num_res_blocks=2,
                dropout=droprate,
                use_attention_at_res=[False, True, True],
                device=device
            )
        else:
            self.unet = ImprovedDenoiseUNet2D(
                in_channels=in_channels,
                out_channels=out_channels,
                model_channels=ch,
                channel_multipliers=[1, 2, 4],
                num_res_blocks=2,
                dropout=droprate,
                use_attention_at_res=[False, True, True],
                device=device
            )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"初始化 DenoiseUNet ({data_dim}): in={in_channels}, out={out_channels}, ch={ch}")
        print(f"总参数量: {total_params:,}")

    def forward(self, x, t, cond=None):
        pred_noise = self.unet(x, t, cond)

        if self.residual:
            return x - pred_noise
        else:
            return pred_noise

    def get_pred_clean(self, x, t, noise_pred, params=None):
        """
        从预测的噪声计算干净信号
        Args:
            x: 带噪信号
            t: 时间步
            noise_pred: 模型预测的噪声
            params: 扩散参数字典，包含 'alphas_cumprod' 等
        """
        if params is None:
            raise ValueError("需要提供扩散参数params")

        # 修复：params 是 dict，使用字典访问方式
        alphas_cumprod = params['alphas_cumprod']
        alpha_bar = alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))

        x_0 = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        return torch.clamp(x_0, -1, 1)


# -----------------------------
# 使用示例
# -----------------------------
if __name__ == "__main__":
    # 测试1D模型
    print("=" * 60)
    print("测试1D模型")
    print("=" * 60)
    model_1d = DenoiseUNet(
        in_channels=1,
        out_channels=1,
        ch=64,
        droprate=0.1,
        device="cuda",
        data_dim='1d'
    )

    batch_size = 4
    signal_length = 5000
    x_1d = torch.randn(batch_size, 1, signal_length)
    t = torch.randint(0, 1000, (batch_size,))

    pred_noise_1d = model_1d(x_1d, t)
    print(f"输入形状: {x_1d.shape}")
    print(f"输出形状: {pred_noise_1d.shape}")

    # 测试2D模型
    print("\n" + "=" * 60)
    print("测试2D模型")
    print("=" * 60)
    model_2d = DenoiseUNet(
        in_channels=1,
        out_channels=1,
        ch=64,
        droprate=0.1,
        device="cuda",
        data_dim='2d'
    )

    x_2d = torch.randn(batch_size, 1, 128, 128)
    pred_noise_2d = model_2d(x_2d, t)
    print(f"输入形状: {x_2d.shape}")
    print(f"输出形状: {pred_noise_2d.shape}")