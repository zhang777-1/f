import ml_collections
from configs import models

def get_config(model="PerceiverEncoder"):
    """获取自编码器（FAE）的配置"""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """基础配置"""
    config = ml_collections.ConfigDict()

    # 随机种子
    config.seed = 42

    # ===============================
    # 大地电磁一维数据输入维度
    # 输入：rho + phase → 2通道
    # 输出：res → 1通道（目标）
    # ===============================
    config.x_dim = [8, 96, 2]      # 改为96，匹配数据补齐长度
    config.coords_dim = [1,]        
    config.y_dim = [8, 96, 1]      # 改为96，匹配数据补齐长度

    # 模式
    config.mode = "train_autoencoder"

    # 项目名称
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_geoelectric_1d"
    wandb.tag = "rho_phase_to_res"

    # 数据集参数
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.num_samples = 1000       # 样本数量（你可以改）
    dataset.num_sensors = 96          # 频率点数量
    dataset.train_batch_size = 32
    dataset.test_batch_size = 32
    dataset.num_workers = 1

    # 学习率
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 1e-6
    lr.peak_value = 8e-5
    lr.decay_rate = 0.95
    lr.transition_steps = 1000
    lr.warmup_steps = 1000

    # 优化器
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5   
    optim.clip_norm = 1.0

    # 训练
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 2000
    training.num_queries = 10
    training.random_sensors = False
    training.sensor_range = [50, 100]
    # ✅ 添加 PDE 开关字段（供命令行使用）
    training.use_pde = False
    

    # 日志与保存
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 100
    logging.eval_interval = 100  # 添加这一行

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 100
    saving.num_keep_ckpts = 3


    return config
