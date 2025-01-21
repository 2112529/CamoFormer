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
                
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(jt.ones(normalized_shape))

    def execute(self, x):
        sigma = x.var(dim=-1, unbiased=False, keepdims=True)
        return x / jt.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.bias = nn.Parameter(jt.zeros(normalized_shape))

    def execute(self, x):
        mu = x.mean(dim=-1, keepdims=True)
        sigma = x.var(dim=-1, unbiased=False, keepdims=True)
        return (x - mu) / jt.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        self.weight.fill_(1.0)
        self.bias.fill_(0.0)

# Utility functions
def to_3d(x):
    b, c, h, w = x.shape
    return x.transpose(0, 2, 3, 1).reshape(b, h * w, c)

def to_4d(x, h, w):
    b, hw, c = x.shape
    return x.reshape(b, h, w, c).transpose(0, 3, 1, 2)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def execute(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv(hidden_features, dim, kernel_size=1, bias=bias)

    def execute(self, x):
        x = self.project_in(x)
        x1, x2 = jt.chunk(self.dwconv(x), 2, dim=1)
        x = nn.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = jt.init.constant((num_heads, 1, 1), "float32", 1.0)

        self.qkv_0 = nn.Conv(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv(dim, dim, kernel_size=1, bias=bias)

    def execute(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        if mask is not None:
            q = q * mask
            k = k * mask

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = q / jt.sqrt(jt.sum(q ** 2, dim=-1, keepdims=True) + 1e-6)
        k = k / jt.sqrt(jt.sum(k ** 2, dim=-1, keepdims=True) + 1e-6)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.temperature
        attn = nn.softmax(attn, dim=-1)

        out = attn @ v
        out = out.reshape(b, -1, h, w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class MSA_head(nn.Module):
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def execute(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MSA_module(nn.Module):
    def __init__(self, dim=128):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head()
        self.F_TA = MSA_head()
        self.TA = MSA_head()
        self.Fuse = nn.Conv(3 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(
            nn.Conv(dim, dim, kernel_size=1),
            nn.Conv(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm(dim),
            nn.ReLU()
        )

    def execute(self, x, side_x, mask):
        N, C, H, W = x.shape
        mask = nn.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)
        mask_d = mask.detach()
        mask_d = jt.sigmoid(mask_d)

        xf = self.F_TA(x * mask_d)
        xb = self.B_TA(x * (1 - mask_d))
        x = self.TA(x)

        x = jt.concat([xb, xf, x], dim=1)
        x = x.reshape(N, 3 * C, H, W)
        x = self.Fuse(x)

        if x.shape[-2:] != side_x.shape[-2:]:
            x = nn.interpolate(x, size=side_x.shape[-2:], mode='bilinear', align_corners=False)

        D = self.Fuse2(side_x + side_x * x)
        return D

    def initialize(self):
        weight_init(self)

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv(in_channels=channels*3, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(num_features=channels)

        self.conv2 = nn.Conv(in_channels=channels, out_channels=channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm(num_features=channels*2)

        self.conv3 = nn.Conv(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm(num_features=channels)

    def execute(self, input1, input2, input3):
        fuse = jt.concat([input1, input2, input3], dim=1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.side_conv1 = nn.Conv(in_channels=512, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv(in_channels=320, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv(in_channels=128, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = Conv_Block(channels)

        self.fuse1 = nn.Sequential(
            nn.Conv(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(channels))
        self.fuse2 = nn.Sequential(
            nn.Conv(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(channels))
        self.fuse3 = nn.Sequential(
            nn.Conv(in_channels=channels*2, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(channels))

        self.MSA5 = MSA_module(dim=channels)
        self.MSA4 = MSA_module(dim=channels)
        self.MSA3 = MSA_module(dim=channels)
        self.MSA2 = MSA_module(dim=channels)

        self.predtrans1 = nn.Conv(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv(in_channels=channels, out_channels=1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv(in_channels=channels, out_channels=1, kernel_size=3, padding=1)

        self.initialize()

    def execute(self, E4, E3, E2, E1, shape):
        E4, E3, E2, E1 = self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)

        if E4.shape[2:] != E3.shape[2:]:
            E4 = nn.interpolate(E4, size=E3.shape[2:], mode='bilinear')
        if E2.shape[2:] != E3.shape[2:]:
            E2 = nn.interpolate(E2, size=E3.shape[2:], mode='bilinear')

        E5 = self.conv_block(E4, E3, E2)

        E4 = jt.concat([E4, E5], dim=1)
        E3 = jt.concat([E3, E5], dim=1)
        E2 = jt.concat([E2, E5], dim=1)

        E4 = nn.relu(self.fuse1(E4))
        E3 = nn.relu(self.fuse2(E3))
        E2 = nn.relu(self.fuse3(E2))

        P5 = self.predtrans5(E5)

        D4 = self.MSA5(E5, E4, P5)
        D4 = nn.interpolate(D4, size=E3.shape[2:], mode='bilinear')
        P4 = self.predtrans4(D4)

        D3 = self.MSA4(D4, E3, P4)
        D3 = nn.interpolate(D3, size=E2.shape[2:], mode='bilinear')
        P3 = self.predtrans3(D3)

        D2 = self.MSA3(D3, E2, P3)
        D2 = nn.interpolate(D2, size=E1.shape[2:], mode='bilinear')
        P2 = self.predtrans2(D2)

        D1 = self.MSA2(D2, E1, P2)
        P1 = self.predtrans1(D1)

        P1 = nn.interpolate(P1, size=shape, mode='bilinear')
        P2 = nn.interpolate(P2, size=shape, mode='bilinear')
        P3 = nn.interpolate(P3, size=shape, mode='bilinear')
        P4 = nn.interpolate(P4, size=shape, mode='bilinear')
        P5 = nn.interpolate(P5, size=shape, mode='bilinear')

        return P5, P4, P3, P2, P1

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            std = math.sqrt(2.0 / m.weight.shape[0])
            nn.init.gauss_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                m.bias.fill_(0.0)
        elif isinstance(m, nn.Conv):
            std = math.sqrt(2.0 / (m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]))
            nn.init.gauss_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                m.bias.fill_(0.0)