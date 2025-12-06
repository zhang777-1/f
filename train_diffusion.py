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



def plot_inversion_result(y_true, y_pred, step, save_path, wandb_log=False, y_min=None, y_max=None, depth_range=(0, 1200), resistivity_range=(0, 200)):
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
    # 加载自编码器的归一化参数
    normalization_path = os.path.join(os.getcwd(), fae_job, "normalization_params.npz")
    if os.path.exists(normalization_path):
        norm_params = np.load(normalization_path)
        y_min = norm_params['y_min']
        y_max = norm_params['y_max']
        print(f"✅ 加载自编码器归一化参数: y_min={y_min:.3f}, y_max={y_max:.3f}")
    else:
        raise FileNotFoundError(f"未找到自编码器归一化参数: {normalization_path}")

    # 为扩散模型生成训练数据（使用与自编码器相同的参数）
    x_train, y_train = generate_dataset(num_samples=config.dataset.num_samples,
    # 数据说明
    # x_train 是坐标数据 (0到1的等间距值)，在所有样本中相同
    # y_train 是目标函数值，在不同样本中变化
    
                                        num_sensors=config.dataset.num_sensors)

    print("使用自编码器的归一化参数...")

    # 使用自编码器的归一化参数对训练数据进行归一化
    x_train_normalized = x_train  # 坐标数据保持原样
    y_train_normalized, _, _ = log_normalize_data(y_train, data_min=y_min, data_max=y_max)

    print(f"✅ 扩散模型训练数据检查:")
    print(f"   x_train: [{x_train_normalized.min():.3f}, {x_train_normalized.max():.3f}] (坐标数据)")
    print(f"   y_train: [{y_train_normalized.min():.3f}, {y_train_normalized.max():.3f}] (使用自编码器归一化)")
    print(f"   使用的归一化范围 - y: [{y_min:.3e}, {y_max:.3e}]")

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
    if os.path.exists(test_data_path):
        print("✅ 加载自编码器的测试集...")
        test_data = np.load(test_data_path)
        x_test = test_data['x_test']
        y_test = test_data['y_test']

        # 使用相同的归一化参数
        normalization_path = os.path.join(os.getcwd(), fae_job, "normalization_params.npz")
        if os.path.exists(normalization_path):
            norm_params = np.load(normalization_path)
            y_min = norm_params['y_min']
            y_max = norm_params['y_max']
            y_test_normalized = log_normalize_data(y_test, data_min=y_min, data_max=y_max)[0]
        else:
            # 如果没有归一化参数，重新归一化
            y_test_normalized, y_min, y_max = log_normalize_data(y_test)
    
        x_test_normalized = x_test  # 坐标数据不需要归一化
    
        print(f"   测试集: x_test{x_test_normalized.shape}, y_test{y_test_normalized.shape}")
        print(f"   归一化参数: y_min={y_min:.3f}, y_max={y_max:.3f}")

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
                if jax.process_index() == 0:  # 只在主进程画图
                    plot_inversion_result(
                        y_true_val, y_pred_val,
                        step,
                        save_path,
                        y_min=y_min,  # 传递归一化参数
                        y_max=y_max,  # 传递归一化参数
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

   