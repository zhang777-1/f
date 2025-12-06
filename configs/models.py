import ml_collections

MODEL_CONFIGS = {}

def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config

@_register
def get_fae_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FAE"   # Function Autoencoder

    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.grid_size = (96,)    # 匹配你的频率点数
    encoder.patch_size = (8,)
    encoder.num_latents = 24     # 输出 [8, 24, 256]
    encoder.emb_dim = 256
    encoder.depth = 6
    encoder.num_heads = 8
    encoder.mlp_ratio = 1
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.fourier_freq = 1.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 2
    decoder.dec_num_heads = 8
    decoder.num_mlp_layers = 2
    decoder.mlp_ratio = 1
    decoder.out_dim = 1          # 输出电阻率
    decoder.layer_norm_eps = 1e-5

    return config

@_register
def get_dit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DiT"

    config.grid_size = (24,)     # 匹配编码器输出的latent数量
    config.emb_dim = 256
    config.depth = 6
    config.num_heads = 8
    config.mlp_ratio = 1
    config.out_dim = 256         # 输出维度匹配隐变量

    # # 条件生成参数
    # config.conditional = conditional = ml_collections.ConfigDict()
    # conditional.use_conditioning = True
    # conditional.condition_dim = 192  # 条件特征维度
    # conditional.condition_proj_dim = 256  # 条件投影维度

    return config

# @_register
# def get_cond_dit_config():
#     """条件DiT配置，专门用于条件生成"""
#     config = ml_collections.ConfigDict()
#     config.model_name = "CondDiT"

#     # 基础DiT参数
#     config.grid_size = (24,)
#     config.emb_dim = 256
#     config.depth = 6
#     config.num_heads = 8
#     config.mlp_ratio = 1
#     config.out_dim = 256

#     # 条件生成增强参数
#     config.conditional = conditional = ml_collections.ConfigDict()
#     conditional.use_conditioning = True
#     conditional.condition_dim = 192
#     conditional.condition_proj_dim = 256
#     conditional.cross_attention_layers = [2, 4]  # 在哪几层加入交叉注意力
#     conditional.condition_dropout = 0.1

#     return config

# @_register
# def get_cond_dit_config():
#     """条件DiT配置"""
#     config = ml_collections.ConfigDict()
#     config.model_name = "CondDiT"  # 使用新名称

#     config.grid_size = (24,)
#     config.emb_dim = 256
#     config.depth = 6
#     config.num_heads = 8
#     config.mlp_ratio = 1
#     config.out_dim = 256
#     config.use_conditioning = True
#     config.condition_dim = 192

#     return config