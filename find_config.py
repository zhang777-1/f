import sys
import os
sys.path.append('.')

try:
    from configs import diffusion
    config = diffusion.get_config('fae,dit')
    print("✅ 成功加载配置")
    print("="*50)
    
    # 打印DiT配置
    if hasattr(config, 'diffusion'):
        print("DiT配置:")
        for key, value in config.diffusion.items():
            print(f"  {key}: {value}")
    else:
        print("❌ 配置中没有diffusion部分")
        
    # 打印自编码器配置
    if hasattr(config, 'autoencoder'):
        print("\n自编码器配置:")
        for key, value in config.autoencoder.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k2, v2 in value.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {key}: {value}")
    
except Exception as e:
    print(f"❌ 加载配置失败: {e}")
    import traceback
    traceback.print_exc()
