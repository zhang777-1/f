import json
import numpy as np
import jax.numpy as jnp

def pad_or_trim(arr, target_len):
    """把每个样本统一为 target_len 长度"""
    result = []
    for sample in arr:
        if len(sample) < target_len:
            # 在末尾用0补齐
            padded = np.pad(sample, (0, target_len - len(sample)), mode='constant')
        else:
            padded = sample[:target_len]
        result.append(padded)
    return np.array(result, dtype=np.float32)

def log_normalize_data(data, data_min=None, data_max=None, eps=1e-8):
    # 确保数据为正数（对数要求）
    data = np.maximum(data, eps)
    
    if data_min is None:
        data_min = np.min(data)
    else:
        data_min = np.maximum(data_min, eps)
        
    if data_max is None:
        data_max = np.max(data)
    else:
        data_max = np.maximum(data_max, eps)
    
    # 确保 data_min < data_max
    data_min = np.minimum(data_min, data_max - eps)
    
    # 应用对数归一化公式
    log_data = np.log10(data)
    log_min = np.log10(data_min)
    log_max = np.log10(data_max)
    
    # 归一化到 [0, 1] 范围
    normalized_data = (log_data - log_min) / (log_max - log_min + eps)
    
    # 确保在 [0, 1] 范围内
    normalized_data = np.clip(normalized_data, 0.0, 1.0)
    
    return normalized_data, data_min, data_max

def log_denormalize_data(normalized_data, data_min, data_max, eps=1e-8):
    # 确保输入在有效范围内
    normalized_data = np.clip(normalized_data, 0.0, 1.0)
    data_min = np.maximum(data_min, eps)
    data_max = np.maximum(data_max, eps)
    
    log_min = np.log10(data_min)
    log_max = np.log10(data_max)
    
    # 反归一化
    log_data = normalized_data * (log_max - log_min) + log_min
    denormalized_data = 10 ** log_data
    
    return denormalized_data

def load_geoelectric_data(json_path, target_len=96, max_samples=1000):
    """通用加载函数，限制样本数量"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def normalize_field(field):
        if isinstance(field, dict):
            field = list(field.values())
        if isinstance(field[0], str):
            field = [json.loads(item) for item in field]
        return np.array(field, dtype=np.float32)

    rho = normalize_field(data['rho'])
    phase = normalize_field(data['phase'])
    res = normalize_field(data['res'])

    # 限制样本数量
    if rho.shape[0] > max_samples:
        rho = rho[:max_samples]
        phase = phase[:max_samples]
        res = res[:max_samples]

    # 统一长度到 target_len
    rho = pad_or_trim(rho, target_len)
    phase = pad_or_trim(phase, target_len)
    res = pad_or_trim(res, target_len)

    x = np.stack([rho, phase], axis=-1)
    y = res[..., np.newaxis]
    
    print(f"调试信息 - 数据形状:")
    print(f"  x: {x.shape}, 范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y: {y.shape}, 范围: [{y.min():.3f}, {y.max():.3f}]")
    
    return jnp.array(x), jnp.array(y)
