from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, normal, xavier_uniform

import flax.linen as nn
from typing import Optional, Callable, Dict, Union, Tuple


# Positional embedding from masked autoencoder https://arxiv.org/abs/2111.06377
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out)  # (M, D/2)
    emb_cos = jnp.cos(out)  # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    pos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        )
    return jnp.expand_dims(pos_embed, 0)


class MlpBlock(nn.Module):
    dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()

    @nn.compact
    def __call__(self, inputs):
        x = nn.Dense(self.dim, kernel_init=self.kernel_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=self.kernel_init)(x)
        return x



class PatchEmbed1D(nn.Module):
    patch_size: tuple = (4,)
    emb_dim: int = 768
    use_norm: bool = False
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, h, c = x.shape

        # 修复：将输入转换为单通道（如果通道数大于1）
        if c != 1:
            x = jnp.mean(x, axis=-1, keepdims=True)  # (batch, h, 2) -> (batch, h, 1)

        # 明确指定卷积参数
        x = nn.Conv(
            features=self.emb_dim,
            kernel_size=(self.patch_size[0],),
            strides=(self.patch_size[0],),
            kernel_init=self.kernel_init,
            name="proj",
        )(x)

        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=self.layer_norm_eps)(x)
        return x




class SelfAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs):
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


class CrossAttnBlock(nn.Module):
    num_heads: int
    emb_dim: int
    mlp_ratio: int
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, q_inputs, kv_inputs):
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(q_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(kv_inputs)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.emb_dim
        )(q, kv)
        x = x + q_inputs
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)

        return x + y


pos_emb_init = get_1d_sincos_pos_embed


class Encoder(nn.Module):
    emb_dim: int
    patch_size: Tuple
    depth: int
    num_heads: int
    mlp_ratio: int
    num_latents: int = 256
    grid_size: Tuple = (100,)
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        b, l, c = x.shape
        x = PatchEmbed1D(self.patch_size, self.emb_dim)(x)

        pos_emb = self.variable(
            "pos_emb",
            "enc_emb",
            get_1d_sincos_pos_embed,
            self.emb_dim,
            l // self.patch_size[0],
        )

        x = x + pos_emb.value

        for _ in range(self.depth):
            x = SelfAttnBlock(
                self.num_heads, self.emb_dim, self.mlp_ratio, self.layer_norm_eps
            )(x)
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        return x


class PerceiverBlock(nn.Module):
    emb_dim: int
    depth: int
    num_heads: int = 8
    num_latents: int = 64
    mlp_ratio: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):  # (B, L,  D) --> (B, L', D)
        latents = self.param('latents',
                             normal(),
                             (self.num_latents, self.emb_dim)  # (L', D)
                             )

        latents = repeat(latents, 'l d -> b l d', b=x.shape[0])  # (B, L', D)
        # Transformer
        for _ in range(self.depth):
            latents = CrossAttnBlock(self.num_heads,
                                     self.emb_dim,
                                     self.mlp_ratio,
                                     self.layer_norm_eps)(latents, x)

        latents = nn.LayerNorm(epsilon=self.layer_norm_eps)(latents)
        return latents


class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Decoder(nn.Module):
    fourier_freq: float = 1.0
    dec_depth: int = 2
    dec_num_heads: int = 8
    dec_emb_dim: int = 256
    mlp_ratio: int = 1
    out_dim: int = 1
    num_mlp_layers: int = 1
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, coords):
        b, n, c = x.shape  # x: (batch, latent_tokens, latent_dim)

        # ========== 修复坐标形状 ==========
        
        # 更健壮的形状处理
        if coords.ndim == 3:
            # 检查是否是 (batch, 1, 96) 需要转置
            if coords.shape[1] == 1 and coords.shape[2] == 96:
                # 转置坐标形状: (batch, 1, 96) -> (batch, 96, 1)
                coords_processed = jnp.transpose(coords, (0, 2, 1))
            else:
                # 其他3D情况直接使用
                coords_processed = coords
        elif coords.ndim == 2:
            if coords.shape[0] == 1 and coords.shape[1] == 96:
                # (1, 96) -> (batch, 96, 1)
                coords_processed = jnp.transpose(coords, (1, 0))  # (96, 1)
                coords_processed = jnp.broadcast_to(coords_processed[None, :, :], (b, 96, 1))
            else:
                # (num_coords, coord_dim) - 广播到batch维度
                coords_processed = jnp.broadcast_to(coords[None, :, :], (b, coords.shape[0], coords.shape[1]))
        elif coords.ndim == 1:
            if coords.shape[0] == 96:
                # (96,) -> (batch, 96, 1)
                coords_processed = jnp.broadcast_to(coords[None, :, None], (b, 96, 1))
            else:
                # 其他1D情况
                coords_processed = jnp.broadcast_to(coords[None, :, None], (b, coords.shape[0], 1))
        else:
            # 默认创建96个坐标点
            coords_processed = jnp.broadcast_to(jnp.linspace(0, 1, 96)[None, :, None], (b, 96, 1))

        # ========== 修复结束 ==========

        # 投影隐变量和坐标到相同维度
        x_proj = nn.Dense(self.dec_emb_dim)(x)  # (batch, n, dec_emb_dim)
        coords_proj = nn.Dense(self.dec_emb_dim)(coords_processed)  # (batch, num_coords, dec_emb_dim)

        # 使用交叉注意力：坐标作为query，隐变量作为key-value
        for _ in range(self.dec_depth):
            coords_proj = CrossAttnBlock(
                num_heads=self.dec_num_heads,
                emb_dim=self.dec_emb_dim,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps
            )(coords_proj, x_proj)  # query=coords, key_value=latents

        # 最终输出处理
        output = nn.LayerNorm(epsilon=self.layer_norm_eps)(coords_proj)
        output = Mlp(
            num_layers=self.num_mlp_layers,
            hidden_dim=self.dec_emb_dim,
            out_dim=self.out_dim,
            layer_norm_eps=self.layer_norm_eps
        )(output)  # (batch, num_coords, out_dim)

        return output


class Mlp(nn.Module):
    num_layers: int
    hidden_dim: int
    out_dim: int
    kernel_init: Callable = xavier_uniform()
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.gelu(x)
        x = nn.Dense(features=self.out_dim)(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    emb_dim: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    emb_dim: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        # Calculate adaLn modulation parameters.
        c = nn.gelu(c)
        c = nn.Dense(6 * self.emb_dim, kernel_init=nn.initializers.constant(0.))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)

        # Attention Residual.
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(kernel_init=nn.initializers.xavier_uniform(),
                                                 num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        # MLP Residual.
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


pos_emb_init = get_1d_sincos_pos_embed

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    model_name: Optional[str]
    grid_size: tuple
    emb_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    out_dim: int

    @nn.compact
    def __call__(self, x, t, c=None):
        b, l, d = x.shape
        
        # 添加输入维度检查和投影
        if d != self.emb_dim:
            print(f"DiT: 输入维度{d} != 期望维度{self.emb_dim}, 进行投影")
            x = nn.Dense(self.emb_dim, name="input_projection")(x)
        
        # 重新获取当前序列长度
        b, l, d = x.shape  # 重新获取投影后的形状
        
        # ========== 修复位置编码 ==========
        # 动态生成位置编码，匹配当前序列长度
        pos_emb = get_1d_sincos_pos_embed(self.emb_dim, l)  # 形状: (1, l, emb_dim)
        pos_emb = jnp.broadcast_to(pos_emb, (b, l, self.emb_dim))  # 广播到batch维度
        # ========== 修复结束 ==========

        x = nn.Dense(self.emb_dim)(x)
        x = x + pos_emb  # 现在形状匹配了

        # 时间嵌入
        t = TimestepEmbedder(self.emb_dim)(t)  # (B, emb_dim)
        
        # 条件处理：如果c不为None，融合条件信息
        if c is not None:
            # 处理条件输入（根据你的数据形状调整）
            if c.ndim == 3:  # [batch, tokens, dim]
                c = jnp.mean(c, axis=1)  # 平均池化
            c_emb = nn.Dense(self.emb_dim)(c)
            t = t + c_emb  # 融合时间和条件

        for _ in range(self.depth):
            x = DiTBlock(self.emb_dim, self.num_heads, self.mlp_ratio)(x, t)
        
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.out_dim)(x)
        return x

