import os
import re

# function_diffusion 目录路径
fd_path = "/public/home/hnust15111216958/zyq/fundiff/fundiff/fundiff/function_diffusion"

print(f"修复 {fd_path} 中的 JAX 导入...")

# 查找所有 Python 文件并修复导入
python_files = []
for root, dirs, files in os.walk(fd_path):
    for file in files:
        if file.endswith('.py'):
            python_files.append(os.path.join(root, file))

fixed_count = 0
for file_path in python_files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含 linear_util 导入
        if 'linear_util' in content:
            # 修复导入
            new_content = content.replace(
                'from jax import linear_util',
                '# from jax import linear_util  # 已注释，使用兼容方案'
            )
            new_content = new_content.replace(
                'import jax.linear_util',
                '# import jax.linear_util  # 已注释，使用兼容方案'
            )
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✅ 修复: {os.path.basename(file_path)}")
                fixed_count += 1
    except Exception as e:
        print(f"❌ 处理 {file_path} 时出错: {e}")

print(f"\n修复完成！处理了 {fixed_count} 个文件")

# 创建兼容性补丁
compatibility_patch = '''
# JAX 兼容性补丁
import jax
import sys

try:
    from jax.interpreters import partial_eval as pe
    # 创建 linear_util 的兼容实现
    class LinearUtilCompat:
        @staticmethod
        def wrap(name, fun):
            return fun
    jax.linear_util = LinearUtilCompat()
    sys.modules['jax.linear_util'] = jax.linear_util
    print("✅ JAX linear_util 兼容层已激活")
except ImportError as e:
    print(f"❌ 创建兼容层失败: {e}")
'''

# 在 function_diffusion 的 __init__.py 中添加补丁
init_file = os.path.join(fd_path, '__init__.py')
if os.path.exists(init_file):
    with open(init_file, 'r', encoding='utf-8') as f:
        init_content = f.read()
    
    if 'linear_util' not in init_content:
        with open(init_file, 'a', encoding='utf-8') as f:
            f.write('\n' + compatibility_patch)
        print("✅ 已添加兼容性补丁到 __init__.py")
else:
    # 创建 __init__.py 如果不存在
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(compatibility_patch)
    print("✅ 已创建包含兼容性补丁的 __init__.py")

print("所有修复完成！")
