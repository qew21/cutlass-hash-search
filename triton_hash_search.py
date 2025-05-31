import triton
import triton.language as tl
import torch
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

@triton.jit
def custom_hash_gpu(val: tl.uint32) -> tl.uint32:
    """GPU优化的哈希函数 (Triton版本)"""
    hash_val = val
    for _ in range(15):  # 保持与原始实现相同的迭代次数
        hash_val = (hash_val ^ 61) ^ (hash_val >> 16)
        hash_val = hash_val + (hash_val << 3)
        hash_val = hash_val ^ (hash_val >> 4)
        hash_val = hash_val * 0x27d4eb2d  # FNV-1a prime
        hash_val = hash_val ^ (hash_val >> 15)
    return hash_val

# --- 高度优化的Triton搜索内核 ---
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=32),
    ],
    key=['SEARCH_SPACE'],
)
@triton.jit
def gpu_brute_force_kernel(
    target_hash: tl.uint32,
    max_val: tl.uint32,
    result_ptr: tl.tensor,
    found_flag_ptr: tl.tensor,
    start_idx: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
    SEARCH_SPACE: tl.constexpr,
):
    """优化的暴力搜索内核"""
    # 计算全局索引
    pid = tl.program_id(0)
    idx = start_idx + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 边界检查
    in_bounds = (idx <= max_val)
    
    # 计算哈希值
    hash_vals = custom_hash_gpu(idx)
    
    # 检查匹配 - 替代tl.any()的实现
    matches = tl.where(in_bounds & (hash_vals == target_hash), idx, 0xFFFFFFFF)
    min_match = tl.min(matches, 0)
    
    # 如果找到匹配
    if min_match != 0xFFFFFFFF:
        # 原子操作确保只有一个线程写入结果
        if tl.atomic_cas(found_flag_ptr, 0, 1) == 0:
            tl.store(result_ptr, min_match)

# --- 包装函数用于启动Triton搜索 ---
def triton_brute_force(target_hash: int, max_val: int, start_idx: int = 0):
    """GPU加速的暴力搜索函数"""
    # 初始化GPU存储
    result = torch.full((1,), 0xFFFFFFFF, dtype=torch.uint32, device='cuda')
    found_flag = torch.zeros(1, dtype=torch.int32, device='cuda')
    
    # 计算搜索空间大小
    search_space = max_val - start_idx + 1
    
    # 定义网格函数
    def grid_fn(meta):
        blocks_needed = (search_space + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']
        return (min(blocks_needed, 65535 * 128),)  # 限制最大块数
    
    # 启动内核
    gpu_brute_force_kernel[grid_fn](
        target_hash,
        max_val,
        result,
        found_flag,
        start_idx,
        # BLOCK_SIZE=None,
        SEARCH_SPACE=search_space,
    )
    
    # 返回结果
    found = result.item()
    return found if found != 0xFFFFFFFF else -1

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

    # GPU (Triton) 搜索
    print("开始 GPU (Triton) 暴力搜索...")
    start_gpu = time.time()
    gpu_result = triton_brute_force(target_hash, max_value_to_check)
    gpu_time = (time.time() - start_gpu) * 1000  # 毫秒
    
    if gpu_result != -1:
        print(f"GPU 找到源值: {gpu_result}")
        print(f"验证哈希: {custom_hash_cpu(gpu_result)}")
    else:
        print("GPU: 在范围内未找到源值")
    print(f"GPU 搜索时间: {gpu_time:.3f} ms\n")

    # 对于小范围可以运行CPU搜索，大范围可能需要注释掉
    if max_value_to_check <= 20_000_000:  
        print("开始 CPU 暴力搜索...")
        start_cpu = time.time()
        cpu_result = cpu_brute_force(target_hash, max_value_to_check)
        cpu_time = (time.time() - start_cpu) * 1000  # 毫秒
        
        if cpu_result != -1:
            print(f"CPU 找到源值: {cpu_result}")
        else:
            print("CPU: 在范围内未找到源值")
        print(f"CPU 搜索时间: {cpu_time:.3f} ms\n")
    else:
        print("跳过 CPU 搜索 (范围过大)")
        cpu_result = -1

    # 结果验证
    if gpu_result != -1:
        if custom_hash_cpu(gpu_result) == target_hash:
            print("验证: GPU 找到的源值哈希匹配")
            if source_to_find == gpu_result:
                print("找到的源值与原始源值完全一致")
            else:
                print("注意: 找到的源值与原始源值不同，可能是哈希碰撞")
        else:
            print("错误: GPU 找到的源值哈希不匹配")