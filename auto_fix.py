import re

# 读取 train_diffusion.py
with open('train_diffusion.py', 'r') as f:
    content = f.read()

# 找到数据归一化部分并替换
old_pattern = r'# 加载归一化参数.*?x_train_combined = np\.concatenate'

new_code = '''# 加载归一化参数
normalization_path = os.path.join(os.getcwd(), fae_job, "normalization_params.npz")

# 强制重新创建归一化参数
if os.path.exists(normalization_path):
    os.remove(normalization_path)

print("创建新的min-max归一化到[-1,1]...")

# 对x和y都进行min-max归一化到[-1,1]
def normalize_to_minus1_1(data):
    data_min = data.min()
    data_max = data.max()
    range_ = data_max - data_min
    range_ = range_ if range_ > 1e-8 else 1.0
    normalized = (data - data_min) / range_ * 2 - 1
    return normalized, data_min, data_max

x_train_normalized, x_min, x_max = normalize_to_minus1_1(x_train)
y_train_normalized, y_min, y_max = normalize_to_minus1_1(y_train)

# 保存归一化参数
if jax.process_index() == 0:
    np.savez(normalization_path, 
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max)

print(f"✅ 归一化检查:")
print(f"   x_train: [{x_train_normalized.min():.3f}, {x_train_normalized.max():.3f}]")
print(f"   y_train: [{y_train_normalized.min():.3f}, {y_train_normalized.max():.3f}]")

# 合并输入
x_train_combined = np.concatenate([x_train_normalized, y_train_normalized], axis=-1)'''

# 使用正则表达式替换
new_content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# 保存修改
with open('train_diffusion.py', 'w') as f:
    f.write(new_content)

print("✅ 自动修复完成！")
print("现在请运行: rm -rf */normalization_params.npz")
print("然后重新开始训练")
