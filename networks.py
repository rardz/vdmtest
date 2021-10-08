import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.serialization import normalize_storage_type

try:
    from torch.nn import SiLU
except:
    # for spport in earlier PyTorch
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
    in_channel,
    out_channel,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
    scale=1,
    mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, time_dim, use_affine_time=False, dropout=0
    ):
        super().__init__()

        self.use_affine_time = use_affine_time
        time_out_dim = out_channel
        time_scale = 1
        norm_affine = True

        if self.use_affine_time:
            time_out_dim *= 2
            time_scale = 1e-10
            norm_affine = False

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = SiLU()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = linear(time_dim, time_out_dim, scale=time_scale)

        self.norm2 = nn.GroupNorm(32, out_channel, affine=norm_affine)
        self.activation2 = SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        if self.use_affine_time:
            gamma, beta = self.time(time).view(batch, -1, 1, 1).chunk(2, dim=1)
            out = (1 + gamma) * self.norm2(out) + beta

        else:
            out = out + self.time(time).view(batch, -1, 1, 1)
            out = self.norm2(out)

        out = self.conv2(self.dropout(self.activation2(out)))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 3, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class TimeLangEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.stack([sinusoid_in.cos(), sinusoid_in.sin()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class TimeFourierEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10, base=2., ):
        super().__init__()

        self.dim = dim

        inv_freq = torch.logspace(0.,
                                  math.log(max_freq / 2) / math.log(base),
                                  dim // 2,
                                  base=base)

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        x = input * 2 - 1  # to [-1, 1]
        shape = x.shape

        sinusoid_in = torch.ger(x.view(-1), self.inv_freq) * math.pi
        pos_emb = torch.stack([sinusoid_in.cos(), sinusoid_in.sin()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb



class ResBlockWithAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        time_dim,
        dropout,
        use_attention=False,
        attention_head=1,
        use_affine_time=False,
    ):
        super().__init__()

        self.resblocks = ResBlock(
            in_channel, out_channel, time_dim, use_affine_time, dropout
        )

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        channel_multiplier,
        n_res_blocks,
        attn_strides,
        attn_heads=1,
        use_affine_time=False,
        dropout=0,
        fold=1,
        cond_src='time',
        max_freq=10,
    ):
        super().__init__()

        self.fold = fold

        time_dim = channel * 4

        self.cond_src = cond_src
        if cond_src == 'time':
            self.time_emb = TimeLangEmbedding(channel)
        elif cond_src == 'eta_norm':
            self.time_emb = TimeFourierEmbedding(channel, max_freq=max_freq)
        else:
            raise ValueError
        self.time_mlp = nn.Sequential(
            linear(channel, time_dim),
            SiLU(),
            linear(time_dim, time_dim),
            SiLU(),
        )

        n_block = len(channel_multiplier)

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                    attention_head=attn_heads,
                    use_affine_time=use_affine_time,
                ),
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_affine_time=use_affine_time,
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                        attention_head=attn_heads,
                        use_affine_time=use_affine_time,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            SiLU(),
            conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10),
        )

    def forward(self, input, time=None, eta_norm=None):
        if self.cond_src == 'time':
            time_embed = self.time_emb(time)
        else: # 'eta_norm'
            time_embed = self.time_emb(eta_norm)
        time_embed = self.time_mlp(time_embed)

        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out(out)
        out = spatial_unfold(out, self.fold)

        return out


class GammaNet(nn.Module):
    """gamma net for negative log SNR
    """
    def __init__(self, d_hidden=1024, gamma_0=-10.0, gamma_1=10.0):
        super().__init__()

        self.w1 = nn.Parameter(torch.zeros(1))
        self.b1 = nn.Parameter(torch.zeros(1))

        self.w2 = nn.Parameter(torch.zeros(d_hidden, 1))
        self.b2 = nn.Parameter(torch.zeros(d_hidden))

        self.w3 = nn.Parameter(torch.zeros(1, d_hidden))
        self.d_hidden = d_hidden

        self._init_weight()
        self.gamma = nn.Parameter(torch.Tensor([gamma_0, gamma_1]))


    def _init_weight(self):
        nn.init.normal_(self.w2, 0, 0.02)
        nn.init.normal_(self.w3, 0, 0.02)

    def forward(self, t, return_be=True):
        # t: b x 1
        if return_be:
            t = torch.cat([torch.Tensor([0, 1]).to(t), t.view(-1)], dim=0).view(-1, 1)

        w1 = F.softplus(self.w1)
        x1 = w1 * t + self.b1

        w2 = F.softplus(self.w2 )
        out = F.linear(x1, weight=w2, bias=self.b2)
        out = torch.sigmoid(out)

        w3 = F.softplus(self.w3)
        out = F.linear(out, weight=w3 / self.d_hidden)

        out = (out + x1).squeeze(-1)
        # to get the soft min, and max
        gamma_0, gamma_1 = -torch.logsumexp(-self.gamma, -1), torch.logsumexp(self.gamma, dim=-1)

        if return_be:
            return out[2:], (out[0], out[1]), (gamma_0, gamma_1)
        return out, None, None


def eta_to_gamma(eta_raw, eta_be, gamma_be, eps=1e-10):
    """
    gamma: [gamma_0, gamma_1]
    """
    (eta_0, eta_1), (gamma_0, gamma_1) = eta_be, gamma_be
    eta_norm = (eta_raw - eta_0) / (eta_1 - eta_0 + eps)
    gamma = (gamma_0 + (gamma_1 - gamma_0) * eta_norm)
    return gamma, eta_norm

def get_eta_scale(eta_be, gamma_be, eps=1e-10):
    (eta_0, eta_1), (gamma_0, gamma_1) = eta_be, gamma_be
    return  (gamma_1 - gamma_0) / (eta_1 - eta_0 + eps)
