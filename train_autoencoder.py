import os
import json
import time
from tqdm import tqdm

from geoelectric_dataset import load_geoelectric_data, log_normalize_data, log_denormalize_data
from einops import repeat

import ml_collections
import wandb

import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)

from model import Encoder, Decoder
from model_utils import create_train_step, create_encoder_step, create_eval_step
from data_utils import generate_dataset, BaseDataset, BatchParser


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    encoder = Encoder(**config.model.encoder)
    decoder = Decoder(**config.model.decoder)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    # print(f"Number of devices: {num_devices}")
    # print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    train_step = create_train_step(encoder, decoder, mesh)
    eval_step = create_eval_step(encoder, decoder, mesh)

    # Create dataloaders
    # x_train, y_train = generate_dataset(num_samples=config.dataset.num_samples,
    #                                     num_sensors=config.dataset.num_sensors)

    # x_train = np.concatenate([x_train, y_train], axis=-1)
    x_train, y_train = load_geoelectric_data('./train_data.json')
    print("x_train:", x_train.shape, "y_train:", y_train.shape)

    # ========== 先归一化整个数据集 ==========
    # 对y进行归一化（x如果是坐标数据可能不需要）
    y_train_normalized, y_min, y_max = log_normalize_data(y_train)
    x_train_normalized = x_train  # 如果x是坐标数据，保持原样

    print(f"归一化后: y范围[{y_train_normalized.min():.3f}, {y_train_normalized.max():.3f}]")
    print(f"归一化参数: y_min={y_min:.3f}, y_max={y_max:.3f}")
    
     # 80/20 数据集划分
    # ========== 再划分训练集验证集 ==========
    total_samples = len(x_train_normalized)
    val_size = total_samples // 5  # 20% 作为验证集
    
    # 划分索引
    indices = np.random.permutation(total_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # 创建临时变量存储划分结果
    x_val_temp = x_train_normalized[val_indices]
    y_val_temp = y_train_normalized[val_indices]
    x_train_temp = x_train_normalized[train_indices]
    y_train_temp = y_train_normalized[train_indices]

    # 重新赋值
    x_val = x_val_temp
    y_val = y_val_temp
    x_train_split = x_train_temp  # 重新赋值训练数据
    y_train_split = y_train_temp  # 重新赋值训练数据

    print(f"从训练数据分割测试集: {val_size}/{total_samples} 样本")
    print(f"训练集: {len(x_train_split)} 样本, 测试集: {len(x_val)} 样本")


    coords = np.linspace(0, 1, config.dataset.num_sensors)[:, None]

     # 创建训练集batch
    batch_coords = repeat(coords, "b d -> n b d", n=jax.device_count())
    batch = (batch_coords, x_train_split, y_train_split)
    batch = jax.tree.map(jnp.array, batch)
    batch = multihost_utils.host_local_array_to_global_array(batch, mesh, P("batch"))

    # 创建验证集batch
    batch_coords_val = repeat(coords, "b d -> n b d", n=jax.device_count())
    batch_val = (batch_coords_val, x_val, y_val)
    batch_val = jax.tree.map(jnp.array, batch_val)
    batch_val = multihost_utils.host_local_array_to_global_array(batch_val, mesh, P("batch"))

    batch = multihost_utils.host_local_array_to_global_array(
        batch, mesh, P("batch")
        )

    # Create checkpoint manager
    job_name = f"{config.model.model_name}"
    job_name += f"_{config.dataset.num_samples}_samples"

    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=job_name, config=config)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Training loop
    rng_key = jax.random.PRNGKey(1234)
    for step in range(config.training.max_steps):
        start_time = time.time()

        state, loss = train_step(state, batch)

        # Logging
        if step % config.logging.log_interval == 0:
            # Log metrics
            train_loss = loss.item()  # 保存训练损失值
            end_time = time.time()
            val_loss = eval_step(state, batch_val)# 然后计算验证损失
            val_loss = val_loss.item()

            log_dict = {"train_loss": train_loss, "val_loss": val_loss, "lr": lr(step)}
            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print("step: {}, train_loss: {:.3e}, val_loss: {:.3e}, time: {:.3e}".format(
            step, train_loss, val_loss, end_time - start_time))

        # Save checkpoint
        if step % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break


    # -------------------
    # 保存测试数据和归一化参数
    # -------------------
    if jax.process_index() == 0:
        print("\n" + "="*50)
        print("保存测试数据和归一化参数")
        print("="*50)
    
        # 1. 保存验证集作为测试集
        test_data_path = os.path.join(os.getcwd(), job_name, "test_data.npz")
        np.savez(test_data_path,
                 x_test=x_val,      # 验证集作为测试集
                 y_test=y_val)
        print(f"✅ 测试集保存到: {test_data_path}")
        print(f"   测试集形状: x_test{x_val.shape}, y_test{y_val.shape}")
    
        # 2. 保存归一化参数
        normalization_path = os.path.join(os.getcwd(), job_name, "normalization_params.npz")
        np.savez(normalization_path, 
                 y_min=y_min, 
                 y_max=y_max)
        print(f"✅ 归一化参数保存到: {normalization_path}")
        print(f"   归一化范围: [{y_min:.3f}, {y_max:.3f}]")
    
        print("="*50)
    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()



