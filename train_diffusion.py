import os
import json
import time

from einops import repeat

import ml_collections
import wandb
import matplotlib.pyplot as plt

import numpy as np

import jax
import jax.numpy as jnp
from jax import random, jit

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    create_diffusion_state,
    compute_total_params,
)
from function_diffusion.utils.train_utils import (
    create_train_diffusion_step,
    get_diffusion_batch,
    sample_ode,
    create_end_to_end_eval_step,
    create_autoencoder_eval_step,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
    restore_fae_state
)

from model import DiT, Encoder, Decoder
from model_utils import create_encoder_step
from data_utils import generate_dataset
from geoelectric_dataset import log_normalize_data, log_denormalize_data
from geoelectric_dataset import load_geoelectric_data


def _plot_inversion_result(y_true, y_pred, step, save_path, wandb_log=False, y_min=None, y_max=None, depth_range=(0, 1200), resistivity_range=(0, 200)):
    """
    绘制反演结果可视化：真实曲线 vs 预测曲线
    """
    # y_true, y_pred shape: (batch, seq_len, 1)
    # 取第一个样本来画
    y_true = np.array(y_true[0, :, 0]) if y_true.ndim == 3 else np.array(y_true[0, :])
    y_pred = np.array(y_pred[0, :, 0]) if y_pred.ndim == 3 else np.array(y_pred[0, :])
    
    # 反归一化处理
    if y_min is not None and y_max is not None:
        y_true = log_denormalize_data(y_true, y_min, y_max)
        y_pred = log_denormalize_data(y_pred, y_min, y_max)
        
    
    # 创建深度坐标
    num_points = len(y_true)
    depths = np.linspace(depth_range[0], depth_range[1], num_points)
    
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(depths, y_true, linewidth=2.5, color='blue', label="True Curve")
    plt.plot(depths, y_pred, linewidth=2.5, linestyle='--', color='red', label="Predicted Curve")

    # 设置坐标轴范围和标签
    plt.xlim(depth_range)
    plt.ylim(resistivity_range)
    
    # 设置自定义刻度
    plt.xticks(np.arange(depth_range[0], depth_range[1] + 1, 400))  # 深度轴刻度
    plt.yticks(np.arange(resistivity_range[0], resistivity_range[1] + 1, 40))  # 电阻率轴刻度
    
    # 设置标签和标题
    plt.xlabel("Depth (m)", fontsize=14)
    plt.ylabel("Resistivity (Ω·m)", fontsize=14)
    plt.title(f"Inversion Result (Step {step})", fontsize=16)
    
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    # 添加数据统计信息
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    plt.text(0.02, 0.98, f'RMSE: {rmse:.2f} Ω·m', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 保存文件
    img_path = os.path.join(save_path, f"inversion_step_{step}.png")
    plt.savefig(img_path, dpi=200)
    plt.close()

    print(f"Saved inversion plot: {img_path}")

    # 上传到 WandB
    if wandb_log:
        wandb.log({"Inversion Visualization": wandb.Image(img_path)}, step=step)


def plot_inversion_result(y_true, y_pred, step, save_path, wandb_log=False, y_min=None, y_max=None, depth_range=(0, 1200), resistivity_range=(0, 200), use_log_scale=False):
    """
    绘制反演结果可视化：真实曲线 vs 预测曲线
    """
    # 数据验证和预处理
    print(f'y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}')
    try:
        # 转换为numpy数组
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # 检查数据有效性
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            print(f"警告：步骤 {step} 的数据包含NaN值，跳过绘图")
            return
            
        if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
            print(f"警告：步骤 {step} 的数据包含无穷值，跳过绘图")
            return
        
        # 处理数据形状：确保是1D数组
        if y_true.ndim == 3:
            y_true = y_true[0, :, 0]  # 取第一个样本的第一个通道
        elif y_true.ndim == 2:
            y_true = y_true[0, :]  # 取第一个样本
        elif y_true.ndim == 1:
            pass  # 已经是1D
        else:
            print(f"警告：y_true的形状 {y_true.shape} 不支持，跳过绘图")
            return
            
        if y_pred.ndim == 3:
            y_pred = y_pred[0, :, 0]  # 取第一个样本的第一个通道
        elif y_pred.ndim == 2:
            y_pred = y_pred[0, :]  # 取第一个样本
        elif y_pred.ndim == 1:
            pass  # 已经是1D
        else:
            print(f"警告：y_pred的形状 {y_pred.shape} 不支持，跳过绘图")
            return
        
        # 检查数据长度是否一致
        if len(y_true) != len(y_pred):
            print(f"警告：y_true长度({len(y_true)})与y_pred长度({len(y_pred)})不一致")
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # 反归一化处理
        if y_min is not None and y_max is not None:
            try:
                y_true = log_denormalize_data(y_true, y_min, y_max)
                y_pred = log_denormalize_data(y_pred, y_min, y_max)
            except Exception as e:
                print(f"反归一化失败: {e}，使用原始数据绘图")
        
        # 检查反归一化后的数据范围
        y_true_range = (np.min(y_true), np.max(y_true))
        y_pred_range = (np.min(y_pred), np.max(y_pred))
        print(f"步骤 {step} - 真实值范围: {y_true_range}, 预测值范围: {y_pred_range}")
        
        # 如果预测值超出合理范围，进行截断
        y_pred_clipped = np.clip(y_pred, -1000, 10000)  # 合理的电阻率范围
        if not np.array_equal(y_pred, y_pred_clipped):
            print(f"警告：预测值超出范围，已进行截断处理")
            y_pred = y_pred_clipped
        
        # 创建深度坐标
        num_points = len(y_true)
        depths = np.linspace(depth_range[0], depth_range[1], num_points)
        
        # 自适应调整绘图范围
        data_min = min(np.min(y_true), np.min(y_pred))
        data_max = max(np.max(y_true), np.max(y_pred))
        
        # 确保数据都是正数（对数刻度要求）
        positive_min = 1e-2  # 最小正值
        y_true_positive = np.clip(y_true, positive_min, None)
        y_pred_positive = np.clip(y_pred, positive_min, None)
        
        # 计算调整后的范围
        if use_log_scale:
            # 对数刻度下的范围计算
            log_min = min(np.log10(np.min(y_true_positive)), np.log10(np.min(y_pred_positive)))
            log_max = max(np.log10(np.max(y_true_positive)), np.log10(np.max(y_pred_positive)))
            
            # 扩大10%的边距
            margin = (log_max - log_min) * 0.1
            log_range = (log_min - margin, log_max + margin)
            
            # 转换回线性刻度
            adjusted_min = 10 ** log_range[0]
            adjusted_max = 10 ** log_range[1]
            resistivity_range = (adjusted_min, adjusted_max)
            print(f"对数刻度调整后绘图范围: {resistivity_range}")
        else:
            # # 线性刻度下的范围计算
            # if data_min < resistivity_range[0] or data_max > resistivity_range[1]:
            #     # 扩大绘图范围以适应数据
            #     margin = (data_max - data_min) * 0.1  # 10%的边距
            #     adjusted_range = (max(0, data_min - margin), data_max + margin)
            #     print(f"线性刻度调整后绘图范围: {adjusted_range}")
            #     resistivity_range = adjusted_range
            min_val = 0
            # 确保范围上限至少为 10000，或略大于数据最大值
            data_max = max(np.max(y_true), np.max(y_pred))
            max_val = max(data_max, 10000) 
            
            # 增加少量边距，防止曲线紧贴边界
            margin = (max_val - min_val) * 0.05
            resistivity_range = (min_val, max_val + margin)
            print(f"线性刻度调整后绘图范围: {resistivity_range}")
        
        plt.figure(figsize=(10, 6))

        # 绘制曲线
        if use_log_scale:
            # 使用对数刻度
            plt.semilogy(depths, y_true_positive, linewidth=2.5, color='blue', label="True Curve")
            plt.semilogy(depths, y_pred_positive, linewidth=2.5, linestyle='--', color='red', label="Predicted Curve")
        else:
            # 使用线性刻度
            plt.plot(depths, y_true, linewidth=2.5, color='blue', label="True Curve")
            plt.plot(depths, y_pred, linewidth=2.5, linestyle='--', color='red', label="Predicted Curve")

        # 设置坐标轴范围和标签
        plt.xlim(depth_range)
        plt.ylim(resistivity_range)
        
        # 设置自定义刻度
        plt.xticks(np.arange(depth_range[0], depth_range[1] + 1, 400))
        
        if use_log_scale:
            # 对数刻度下的智能刻度
            from matplotlib.ticker import LogLocator, ScalarFormatter
            plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=10))
            plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
        else:
            # # 线性刻度下的智能刻度
            # y_range = resistivity_range[1] - resistivity_range[0]
            # if y_range <= 5:
            #     y_tick_step = 0.5
            # elif y_range <= 20:
            #     y_tick_step = 2
            # elif y_range <= 100:
            #     y_tick_step = 10
            # else:
            #     y_tick_step = 20
            
            # plt.yticks(np.arange(resistivity_range[0], resistivity_range[1] + 1, y_tick_step))
            custom_yticks = np.arange(0, 10001, 1000) # 生成 0, 1000, ..., 10000
    
            # 检查自定义刻度是否在调整后的绘图范围内
            if custom_yticks[-1] <= resistivity_range[1]:
                plt.yticks(custom_yticks)
            else:
                # 如果自定义刻度超出范围（例如，数据最大值超过 10000，但刻度只到 10000），
                # 则使用自定义列表，但图可能会裁切
                plt.yticks(custom_yticks)
                print("注意：实际绘图上限可能高于10000，但刻度仍设定为 10000。")
        
        # 设置标签和标题
        plt.xlabel("Depth (m)", fontsize=14)
        plt.ylabel("Resistivity (Ω·m)", fontsize=14)
        plt.title(f"Inversion Result (Step {step})", fontsize=16)
        
        plt.grid(alpha=0.3, which='both')  # 显示所有网格线
        plt.legend(fontsize=12)

        # 添加数据统计信息
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        plt.text(0.02, 0.98, f'RMSE: {rmse:.2f} Ω·m', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 保存文件
        img_path = os.path.join(save_path, f"inversion_step_{step}.png")
        plt.savefig(img_path, dpi=200)
        plt.close()

        print(f"Saved inversion plot: {img_path}")

        # 上传到 WandB
        if wandb_log:
            wandb.log({"Inversion Visualization": wandb.Image(img_path)}, step=step)
            
    except Exception as e:
        print(f"绘图过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def train_and_evaluate(config: ml_collections.ConfigDict):
    # -------------------
    # 1) Initialize autoencoder (load checkpoint if available)
    # -------------------
    encoder = Encoder(**config.autoencoder.encoder)
    decoder = Decoder(**config.autoencoder.decoder)
    fae_job = f"{config.autoencoder.model_name}" + f"_{config.dataset.num_samples}_samples"

    # Try to restore fae_state from checkpoint. If not found, initialize fresh state.
    try:
        fae_state = restore_fae_state(config, fae_job, encoder, decoder)
        print("Loaded FAE state from checkpoint.")
    except Exception as e:
        print("Could not restore FAE state from checkpoint (will initialize fresh). Error:", e)
        # Create optimizer for fae initialization (reusing create_autoencoder_state API)
        fae_state = create_autoencoder_state(config, encoder, decoder, create_optimizer(config)[1])

    # -------------------
    # 2) Initialize diffusion model (do not overwrite saved checkpoints)
    # -------------------
    use_conditioning = False
    diffusion_config = dict(config.diffusion)
    dit_supported_params = [
        'grid_size', 'emb_dim', 'depth', 'num_heads',
        'mlp_ratio', 'out_dim'
    ]
    filtered_config = {k: v for k, v in diffusion_config.items() if k in dit_supported_params}
    dit = DiT(model_name=config.diffusion.model_name, **filtered_config)

    lr, tx = create_optimizer(config)

    # Try to restore diffusion state from checkpoint if available, otherwise initialize
    job_name = f"{config.diffusion.model_name}_{config.dataset.num_samples}_samples"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)
    try:
        state = restore_checkpoint(ckpt_mngr, None)  # if implementation returns latest
        if state is None:
            raise RuntimeError("No diffusion checkpoint found - will init new state")
        print("Loaded diffusion state from checkpoint.")
    except Exception:
        print("Initializing new diffusion state.")
        state = create_diffusion_state(config, dit, tx, use_conditioning=use_conditioning)

    # 强制转换state为TrainState（如果是字典）
    if isinstance(state, dict) and 'params' in state:
        from flax.training import train_state
        state = train_state.TrainState.create(
            apply_fn=dit.apply,
            params=state['params'],
            tx=tx
        )

    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # -------------------
    # 3) Device / sharding
    # -------------------
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}, local: {num_local_devices}")

    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())
    fae_state = multihost_utils.host_local_array_to_global_array(fae_state, mesh, P())

    # train / encoder steps
    train_step = create_train_diffusion_step(dit, mesh, use_conditioning=False)
    encoder_step = create_encoder_step(encoder, mesh)

    # 创建保存路径用于存储评估图像
    save_path = os.path.join(os.getcwd(), "evaluation_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # -------------------
    # 4) Dataset and coords - KEEP CONSISTENT WITH FAE TRAINING
    # -------------------
    
    # 为扩散模型生成训练数据
    # 数据说明
    # x_train 是坐标数据 (0到1的等间距值)，在所有样本中相同
    # y_train 是目标函数值，在不同样本中变化
    x_train, y_train = load_geoelectric_data('./train_data.json')
    # 自动计算归一化参数
    # 首先尝试加载已保存的归一化参数
    # 从训练数据中计算归一化参数
    y_min = float(y_train.min())
    y_max = float(y_train.max())
    print(f" 计算归一化参数: y_min={y_min:.3f}, y_max={y_max:.3f}")

    # 使用计算的归一化参数对训练数据进行归一化
    x_train_normalized = x_train  # 坐标数据保持原样
    y_train_normalized, _, _ = log_normalize_data(y_train, data_min=y_min, data_max=y_max)

    print(f"✅ 扩散模型训练数据检查:")
    # print(f"   x_train: [{x_train_normalized.min():.3f}, {x_train_normalized.max():.3f}] (坐标数据)")
    print(f"   y_train: [{y_train_normalized.min():.3f}, {y_train_normalized.max():.3f}] (归一化后)")
    print(f"   原始数据范围 - y: [{y_min:.3e}, {y_max:.3e}]")
    # print(f"   归一化参数保存路径: {normalization_path}")

    # 合并输入（保持与自编码器训练一致）
    condition_data = x_train_normalized  # 只包含坐标数据作为条件

    # IMPORTANT: coords must match the shape used when training the autoencoder.
    # Typically coords = (num_sensors, 1)
    coords = np.linspace(0, 1, config.dataset.num_sensors)[:, None]  # shape (num_sensors, 1)

    # Repeat coords across devices: shape (n_devices, num_sensors, 1)
    batch_coords = repeat(coords, "b d -> n b d", n=jax.device_count())

    batch = (batch_coords, condition_data, y_train_normalized) 
    batch = jax.tree.map(jnp.array, batch)
    batch = multihost_utils.host_local_array_to_global_array(batch, mesh, P("batch"))

    # If checkpoint dir doesn't exist, create and save config / wandb init
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        # save config
        config_dict = config.to_dict()
        with open(os.path.join(os.getcwd(), job_name, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
        # init wandb
        try:
            wandb.init(project=config.wandb.project, name=job_name, config=config)
        except Exception as e:
            print("WandB init failed (continuing):", e)

    # Ensure ckpt manager exists
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # -------------------
    # 5) Prepare test set (keep shapes consistent)
    # -------------------
    test_data_path = os.path.join(os.getcwd(), fae_job, "test_data.npz")
    normalization_path = os.path.join(os.getcwd(), fae_job, "normalization_params.npz") 
    print(f"Looking for test data at: {test_data_path}")
    print(f"Looking for normalization params at: {normalization_path}")
    if os.path.exists(test_data_path) and os.path.exists(normalization_path):
        print("✅ 加载自编码器的测试集和归一化参数...")
        print(f"Looking for test data at: {test_data_path}")
        test_data = np.load(test_data_path)
        x_test = test_data['x_test']
        y_test = test_data['y_test']

        norm_params = np.load(normalization_path)  
        y_min = float(norm_params['y_min'])
        y_max = float(norm_params['y_max'])
    
        print(f"   测试集形状: x_test{x_test.shape}, y_test{y_test.shape}")
        print(f"   测试数据范围: [{y_test.min():.3f}, {y_test.max():.3f}]")
        print(f"   归一化参数: y_min={y_min:.3f}, y_max={y_max:.3f}")
        # 验证数据是否确实是归一化的
        if y_test.min() < -10 or y_test.max() > 10:
            print(f"⚠️  警告：测试数据范围异常 [{y_test.min():.3f}, {y_test.max():.3f}]")
            print("   数据可能未正确归一化或使用了不同的归一化方式")
    else:
        # 如果没有测试数据文件，从训练数据中分割一部分作为测试集
        print("⚠️ 未找到测试数据文件，从训练数据中分割20%作为测试集...")
        split_idx = int(0.8 * len(x_train))
        
        # 注意：这里分割的是原始数据，而不是归一化后的数据
        x_test = x_train[split_idx:]
        y_test = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # 重新计算测试数据的归一化参数（基于原始数据）
        y_test_min = float(y_test.min())
        y_test_max = float(y_test.max())
        
        # 使用测试数据自身的归一化参数进行归一化
        y_test_normalized = log_normalize_data(y_test, data_min=y_test_min, data_max=y_test_max)[0]
        x_test_normalized = x_test  # 坐标数据不需要归一化
        
        # 重新归一化训练数据
        y_train_normalized = log_normalize_data(y_train, data_min=y_min, data_max=y_max)[0]
        x_train_normalized = x_train
        
        print(f"   训练集: x_train{x_train_normalized.shape}, y_train{y_train_normalized.shape}")
        print(f"   测试集: x_test{x_test_normalized.shape}, y_test{y_test_normalized.shape}")
        print(f"   测试数据归一化参数: y_min={y_test_min:.3f}, y_max={y_test_max:.3f}")

    # 使用归一化后的数据
    condition_data_test = x_test_normalized  # 只包含坐标数据作为条件

    batch_coords_test = repeat(coords, "b d -> n b d", n=jax.device_count())
    test_batch = (batch_coords_test, condition_data_test, y_test_normalized)
    test_batch = jax.tree.map(jnp.array, test_batch)
    test_batch = multihost_utils.host_local_array_to_global_array(test_batch, mesh, P("batch"))

    # -------------------
    # 6) End-to-end eval step
    # -------------------
    end_to_end_eval_step = create_end_to_end_eval_step(encoder, decoder, dit, mesh, use_conditioning=False)
    autoencoder_eval_step = create_autoencoder_eval_step(encoder, decoder, mesh)

    # -------------------
    # 7) Training loop
    # -------------------
    rng = random.PRNGKey(config.training.seed if 'seed' in config.training else 0)
    for step in range(config.training.max_steps):
        start_time = time.time()
        rng, _ = random.split(rng)

        z_u = encoder_step(fae_state.params[0], batch) 

        diff_batch, rng = get_diffusion_batch(rng, z1=z_u, c=None, use_conditioning=False)
        state, loss = train_step(state, diff_batch)

        # Logging
        if step % config.logging.log_interval == 0:
            loss_val = float(loss)
            end_time = time.time()
            log_dict = {"loss": loss_val, "lr": float(lr(step)) if callable(lr) else lr}
            if jax.process_index() == 0:
                try:
                    wandb.log(log_dict, step)
                except Exception:
                    pass
                print(f"step: {step}, loss: {loss_val:.3e}, time: {end_time - start_time:.3f}")

        # Periodic end-to-end evaluation
        if step % config.logging.eval_interval == 0:
            try:
                # 现在返回两个值：rmse, normalized_rmse
                rmse_val, normalized_rmse_val, y_pred_val, y_true_val = end_to_end_eval_step(
                    fae_state, state, test_batch
                )

                # print(f'pred_res = {y_pred_val}')
                # print(f'true_res = {y_true_val}')

                if jax.process_index() == 0:  # 只在主进程画图
                    plot_inversion_result(
                        y_true_val, y_pred_val,
                        step,
                        save_path,
                        y_min=y_min,  # 使用测试数据的归一化参数
                        y_max=y_max,  # 使用测试数据的归一化参数
                        wandb_log=True
                    )
                rmse_val = float(rmse_val) if rmse_val is not None else None
                normalized_rmse_val = float(normalized_rmse_val) if normalized_rmse_val is not None else None
            except Exception as e:
                rmse_val, normalized_rmse_val = None, None
                print("End-to-end eval failed:", e)
                import traceback
                traceback.print_exc()
            
            if jax.process_index() == 0:
                if rmse_val is not None and normalized_rmse_val is not None:
                    print(f"step: {step}, diffusion_loss: {float(loss):.3e}, end_to_end_rmse: {rmse_val:.3e}, normalized_rmse: {normalized_rmse_val:.3f}")
                    try:
                        wandb.log({
                            "end_to_end_rmse": rmse_val,
                            "normalized_end_to_end_rmse": normalized_rmse_val
                        }, step)
                    except Exception:
                        pass
                else:
                    print(f"step: {step}, diffusion_loss: {float(loss):.3e}, end_to_end_loss: N/A")

        # Save checkpoint at intervals
        if step % config.saving.save_interval == 0:
            if jax.process_index() == 0:
                print(f"saving checkpoint at step {step}...")
            save_checkpoint(ckpt_mngr, state)

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()

    # -------------------
    # 8) Unified evaluation (use test set, shapes kept consistent)
    # -------------------
    # -------------------
    # 8) Unified evaluation (use test set, shapes kept consistent)
    # -------------------
    print("\n" + "="*50)
    print("开始统一模型评估")
    print("="*50)

    print("1. 评估自编码器重建性能...")
    try:
        # 现在返回两个值：rmse, normalized_rmse
        autoencoder_rmse, autoencoder_normalized_rmse = autoencoder_eval_step(fae_state, test_batch)
        autoencoder_rmse = float(autoencoder_rmse) if autoencoder_rmse is not None else None
        autoencoder_normalized_rmse = float(autoencoder_normalized_rmse) if autoencoder_normalized_rmse is not None else None
    except Exception as e:
        autoencoder_rmse, autoencoder_normalized_rmse = None, None
        print("Autoencoder eval failed:", e)

    print("2. 评估扩散模型生成性能...")
    try:
        # 现在返回四个值：rmse, normalized_rmse
        diffusion_rmse, diffusion_normalized_rmse, _, _ = end_to_end_eval_step(fae_state, state, test_batch)
        diffusion_rmse = float(diffusion_rmse) if diffusion_rmse is not None else None
        diffusion_normalized_rmse = float(diffusion_normalized_rmse) if diffusion_normalized_rmse is not None else None
    except Exception as e:
        diffusion_rmse, diffusion_normalized_rmse = None, None
        print("End-to-end diffusion eval failed:", e)

    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)

    if jax.process_index() == 0:
        if autoencoder_rmse is not None and autoencoder_normalized_rmse is not None:
            print(f"自编码器 - RMSE: {autoencoder_rmse:.6f}, NRMSE: {autoencoder_normalized_rmse:.6f} ({autoencoder_normalized_rmse*100:.1f}%)")
        else:
            print("自编码器测试: 评估失败")
        
        if diffusion_rmse is not None and diffusion_normalized_rmse is not None:
            print(f"扩散模型端到端 - RMSE: {diffusion_rmse:.6f}, NRMSE: {diffusion_normalized_rmse:.6f} ({diffusion_normalized_rmse*100:.1f}%)")
        else:
            print("扩散模型端到端: 评估失败")

        if autoencoder_normalized_rmse is not None and diffusion_normalized_rmse is not None:
            print(f"性能对比: 扩散模型比自编码器 {'更好' if diffusion_normalized_rmse < autoencoder_normalized_rmse else '稍差'}")
            try:
                wandb.log({
                    "final_autoencoder_rmse": autoencoder_rmse,
                    "final_autoencoder_nrmse": autoencoder_normalized_rmse,
                    "final_diffusion_rmse": diffusion_rmse,
                    "final_diffusion_nrmse": diffusion_normalized_rmse,
                    "performance_gap": diffusion_normalized_rmse - autoencoder_normalized_rmse
                }, step=config.training.max_steps)
            except Exception:
                pass

    print("所有模型训练和评估完成！")
    print("="*50)

   
