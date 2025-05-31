import time
import random
import numpy as np

# --- 优化的哈希函数 (与原始C++版本兼容) ---
def custom_hash_cpu(val: int) -> int:
    """CPU版本的哈希函数，用于生成目标哈希值"""
    val = np.uint32(val)
    for _ in range(15):
        val = (val ^ np.uint32(61)) ^ (val >> np.uint32(16))
        val = val + (val << np.uint32(3))
        val = val ^ (val >> np.uint32(4))
        val = np.uint32(val * np.uint32(0x27d4eb2d))
        val = val ^ (val >> np.uint32(15))
    return int(val)


# --- CPU参考实现 ---
def cpu_brute_force(target_hash: int, max_val: int) -> int:
    """CPU暴力搜索实现"""
    for i in range(max_val + 1):
        if custom_hash_cpu(i) == target_hash:
            return i
    return -1

# --- 主程序 ---
if __name__ == "__main__":
    # 配置参数
    max_value_to_check = 2 * 1000 * 1000 * 1000  # 20亿
    source_to_find = 142385552
    
    # 计算目标哈希值
    target_hash = custom_hash_cpu(source_to_find)
    
    print(f"目标哈希值: {target_hash}")
    print(f"需要查找的源值: {source_to_find}")
    print(f"最大检查值: {max_value_to_check}\n")

    print("开始 CPU 暴力搜索...")
    start_cpu = time.time()
    cpu_result = cpu_brute_force(target_hash, max_value_to_check)
    cpu_time = (time.time() - start_cpu) * 1000  # 毫秒
    
    if cpu_result != -1:
        print(f"CPU 找到源值: {cpu_result}")
    else:
        print("CPU: 在范围内未找到源值")
    print(f"CPU 搜索时间: {cpu_time:.3f} ms\n")
