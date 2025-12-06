import re

print("查找sample_ode调用和num_steps参数...")
print("="*60)

# 检查train_diffusion.py
with open('train_diffusion.py', 'r') as f:
    content = f.read()
    
# 查找sample_ode调用
matches = re.findall(r'sample_ode\([^)]+\)', content)
if matches:
    print("train_diffusion.py中的sample_ode调用:")
    for match in matches[:3]:  # 最多显示3个
        print(f"  {match}")
else:
    print("train_diffusion.py中未找到sample_ode调用")
    
# 查找num_steps参数
num_steps_matches = re.findall(r'num_steps\s*=\s*(\d+)', content)
if num_steps_matches:
    print(f"\ntrain_diffusion.py中的num_steps值: {num_steps_matches}")
else:
    print("\ntrain_diffusion.py中未找到num_steps参数")

print("\n" + "="*60)

# 检查model_utils.py
try:
    with open('model_utils.py', 'r') as f:
        content = f.read()
    
    # 查找sample_ode函数定义
    func_match = re.search(r'def sample_ode\([^)]+\):(.*?)(?=\n\S|\Z)', content, re.DOTALL)
    if func_match:
        print("model_utils.py中的sample_ode函数:")
        print(func_match.group(0)[:200] + "...")
        
        # 查找num_steps默认值
        default_match = re.search(r'num_steps\s*=\s*(\w+)', func_match.group(0))
        if default_match:
            print(f"num_steps默认值: {default_match.group(1)}")
        else:
            print("未找到num_steps默认值")
    else:
        print("model_utils.py中未找到sample_ode函数")
        
except FileNotFoundError:
    print("model_utils.py文件不存在")

print("\n" + "="*60)
