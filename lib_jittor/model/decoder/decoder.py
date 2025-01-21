import jittor as jt
from jittor import nn
import math
from einops import rearrange
import numbers

# Weight initialization function
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.Identity)):
            pass
        else:
            if hasattr(m, 'initialize'):
                m.initialize()

# Utility functions
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.norm(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv(dim, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2)
        self.project_out = nn.Conv(hidden_features, dim, kernel_size=1)

    def execute(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = nn.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(jt.ones(num_heads, 1, 1))

        self.qkv_conv = nn.Conv(dim, dim * 3, kernel_size=1)
        self.project_out = nn.Conv(dim, dim, kernel_size=1)

    def execute(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (head c) h w -> b head c (h w)', head=self.num_heads), qkv)

        q = nn.normalize(q, dim=-1)
        k = nn.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.softmax(attn, dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class MSA_head(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor)

    def execute(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MSA_module(nn.Module):
    def __init__(self, dim):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head(dim, num_heads=8, ffn_expansion_factor=4)
        self.F_TA = MSA_head(dim, num_heads=8, ffn_expansion_factor=4)
        self.TA = MSA_head(dim, num_heads=8, ffn_expansion_factor=4)
        self.Fuse = nn.Conv(3 * dim, dim, kernel_size=3, padding=1)

    def execute(self, x, side_x, mask):
        mask = nn.interpolate(mask, size=x.shape[2:], mode='bilinear')
        mask = nn.Sigmoid(mask)

        xf = self.F_TA(x * mask)
        xb = self.B_TA(x * (1 - mask))
        xt = self.TA(x)

        x = jt.concat([xb, xf, xt], dim=1)
        x = self.Fuse(x)
        D = x * side_x + side_x
        return D

class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.side_conv1 = nn.Conv(512, channels, kernel_size=3, padding=1)
        self.side_conv2 = nn.Conv(320, channels, kernel_size=3, padding=1)
        self.side_conv3 = nn.Conv(128, channels, kernel_size=3, padding=1)
        self.side_conv4 = nn.Conv(64, channels, kernel_size=3, padding=1)

        self.MSA5 = MSA_module(dim=channels)
        self.MSA4 = MSA_module(dim=channels)
        self.MSA3 = MSA_module(dim=channels)
        self.MSA2 = MSA_module(dim=channels)

        self.predtrans = nn.Conv(channels, 1, kernel_size=3, padding=1)
        weight_init(self)

    def execute(self, E4, E3, E2, E1, shape):
        E4 = self.side_conv1(E4)
        E3 = self.side_conv2(E3)
        E2 = self.side_conv3(E2)
        E1 = self.side_conv4(E1)

        P5 = self.MSA5(E4, E3, E4)
        P4 = self.MSA4(E3, E2, P5)
        P3 = self.MSA3(E2, E1, P4)
        P2 = self.MSA2(E1, E1, P3)

        P1 = self.predtrans(P2)
        return nn.interpolate(P1, size=shape, mode='bilinear')
