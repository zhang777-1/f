import ml_collections
from configs import models

def get_config(autoencoder_diffusion):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()

    autoencoder, diffusion = autoencoder_diffusion.split(',')

    get_autoencoder_config = getattr(models, f"get_{autoencoder}_config")
    get_diffusion_config = getattr(models, f"get_{diffusion}_config")

    config.autoencoder = get_autoencoder_config()
    config.diffusion = get_diffusion_config()
    return config

def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # ===============================
    # 维度配置 - 针对大地电磁数据
    # ===============================
    # 输入：视电阻率+相位 [batch, 96, 2]
    config.x_dim = [8, 96, 2]      
    # 条件：视电阻率+相位 [batch, 96, 2]  
    config.c_dim = [8, 96, 2]      
    # 隐变量维度（从编码器输出）
    config.z_dim = [8, 24, 256]    # 根据你的编码器输出调整
    # 时间步维度
    config.t_dim = [8,]            
    # 坐标维度
    config.coords_dim = [1,]

    # 训练模式
    config.mode = "train_diffusion"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_geoelectric_1d"
    wandb.tag = "conditional_diffusion"

    # 条件生成配置
    # config.conditional = conditional = ml_collections.ConfigDict()
    # conditional.use_conditioning = True  # 启用条件生成
    # conditional.condition_type = "concat"  # 条件连接方式
    # conditional.condition_dim = 192  # 条件特征维度 96*2

     # 条件生成配置
    config.use_conditioning = False
    config.c_dim = [8, 96, 2]  # 条件输入维度


    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.num_samples = 1000
    dataset.num_sensors = 96
    dataset.train_batch_size = 32
    dataset.test_batch_size = 32
    dataset.eval_batch_size = 4
    dataset.num_workers = 1
    dataset.condition_channels = 2  # 视电阻率+相位
    dataset.target_channels = 1     # 电阻率

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 1e-6
    lr.peak_value = 8e-5
    lr.decay_rate = 0.95
    lr.transition_steps = 1000
    lr.warmup_steps = 1000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 2000
    training.use_conditioning = False  # 使用条件输入

    # 扩散过程参数
    config.diffusion_params = diffusion_params = ml_collections.ConfigDict()
    diffusion_params.num_steps = 2000
    diffusion_params.beta_schedule = "linear"
    diffusion_params.prediction_type = "epsilon"  # 预测噪声

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 100
    logging.eval_interval = 200  # 添加这一行

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 100
    saving.num_keep_ckpts = 3

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.num_samples = 200
    eval.num_steps = 200
    eval.batch_size = 16

    return config