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

def _log_normalize_data(data, data_min=None, data_max=None, eps=1e-8):
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

def log_normalize_data(data, data_min=None, data_max=None, eps=1e-8):
    """
    根据数据范围选择合适的归一化方法
    如果数据最大值 <= 10，只进行min-max归一化
    如果数据最大值 > 10，先取log10再进行min-max归一化
    """
    # 确保数据为正数
    data = np.maximum(data, eps)
    
    # 如果未提供data_max，计算实际最大值
    if data_max is None:
        actual_max = np.max(data)
    else:
        actual_max = data_max
    
    print(f"[log_normalize_data] 数据最大值: {actual_max:.3f}")
    
    # 判断数据范围并选择归一化方法
    if actual_max <= 10:
        print(f"[log_normalize_data] 选择min-max归一化 (最大值 <= 10)")
        # 只进行min-max归一化
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
        
        # 归一化到 [0, 1] 范围
        normalized_data = (data - data_min) / (data_max - data_min + eps)
        
        # 确保在 [0, 1] 范围内
        normalized_data = np.clip(normalized_data, 0.0, 1.0)
        
        print(f"[log_normalize_data] min-max归一化完成，data_min: {data_min:.3f}, data_max: {data_max:.3f}")
        return normalized_data, data_min, data_max
    else:
        print(f"[log_normalize_data] 选择对数归一化 (最大值 > 10)")
        # 数据最大值 > 10，使用对数归一化
        result = _log_normalize_data(data, data_min, data_max, eps)
        print(f"[log_normalize_data] 对数归一化完成，data_min: {result[1]:.3f}, data_max: {result[2]:.3f}")
        return result

def _log_denormalize_data(normalized_data, data_min, data_max, eps=1e-8):
    """
    根据data_max参数值选择不同的反归一化方式
    
    如果data_max <= 1: 先进行min-max反归一化到1-4，再计算10^x
    如果data_max <= 4: 直接计算10^x
    """
    # 确保输入在有效范围内
    # normalized_data = np.clip(normalized_data, 0.0, 1.0)
    data_min = np.maximum(data_min, eps)
    data_max = np.maximum(data_max, eps)
    
    print(f"[log_denormalize_data] data_min: {data_min:.3f}, data_max: {data_max:.3f}")
    
    if data_max <= 1:
        print(f"[log_denormalize_data] data_max <= 1，先反归一化到1-4范围再计算10^x")
        # 先反归一化到1-4范围
        target_min = 1.0
        target_max = 4.0
        log_data = normalized_data * (target_max - target_min) + target_min
        denormalized_data = 10 ** log_data
        print(f"[log_denormalize_data] 反归一化到范围 [{target_min}, {target_max}]，结果范围: [{denormalized_data.min():.3f}, {denormalized_data.max():.3f}]")
    elif data_max <= 4:
        print(f"[log_denormalize_data] data_max <= 4，直接计算10^x")
        # 直接计算10^x
        log_min = np.log10(data_min)
        log_max = np.log10(data_max)
        log_data = normalized_data * (log_max - log_min) + log_min
        denormalized_data = 10 ** log_data
        print(f"[log_denormalize_data] 直接计算10^x，结果范围: [{denormalized_data.min():.3f}, {denormalized_data.max():.3f}]")
    else:
        print(f"[log_denormalize_data] data_max > 4，使用原始的log反归一化")
        # data_max > 4，使用原始的log反归一化
        log_min = np.log10(data_min)
        log_max = np.log10(data_max)
        log_data = normalized_data * (log_max - log_min) + log_min
        denormalized_data = 10 ** log_data
        print(f"[log_denormalize_data] 使用log反归一化，结果范围: [{denormalized_data.min():.3f}, {denormalized_data.max():.3f}]")
    
    return denormalized_data

def log_denormalize_data(normalized_data, data_min, data_max, eps=1e-8):
    """
    根据data_max参数值，选择反转log或min-max
    """
    # 确保 data_min/max 为正
    data_min = np.maximum(data_min, eps)
    data_max = np.maximum(data_max, eps)
    
    # 严格匹配归一化时的分界点：10
    if data_max <= 10:
        print(f"[log_denormalize_data] 逆转min-max归一化 (data_max <= 10)")
        # 归一化时是线性 Min-Max: normalized = (data - data_min) / (data_max - data_min)
        # 逆操作: data = normalized * (data_max - data_min) + data_min
        denormalized_data = (normalized_data * (data_max - data_min)) + data_min
        
    else: # data_max > 10
        print(f"[log_denormalize_data] 逆转对数归一化 (data_max > 10)")
        # 归一化时是 Log10 -> Min-Max
        log_min = np.log10(data_min)
        log_max = np.log10(data_max)
        
        # 逆转 Min-Max: log_data = normalized * (log_max - log_min) + log_min
        log_data = normalized_data * (log_max - log_min) + log_min
        
        # 逆转 Log10: data = 10 ** log_data
        denormalized_data = 10 ** log_data
        
    print(f"[log_denormalize_data] 反归一化完成，结果范围: [{denormalized_data.min():.3f}, {denormalized_data.max():.3f}]")
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
    print(f"\n{'='*60}")
    print(f"加载原始数据检查")
    print(f"{'='*60}")
    print(f"res (从JSON加载):")
    print(f"  范围: [{res.min():.3f}, {res.max():.3f}]")
    # 检测并转换 log 值为原始值
    if res.min() >= 0 and res.max() <= 5:
        print(f"  检测到 res 是 log10 值，转换为原始电阻率...")
        res = 10 ** res
        print(f"  转换后: [{res.min():.1f}, {res.max():.1f}] Ω·m")
    
    # rho 保持原样（已经是原始值）
    print(f"\nrho (视电阻率): [{rho.min():.1f}, {rho.max():.1f}] Ω·m")
    print(f"{'='*60}\n")

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

