import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

# --------------------------------------------------
# CUDA kernel: 与你给出的 C++ 版本一致
# --------------------------------------------------
mod = SourceModule(r"""
__device__ unsigned int custom_hash(unsigned int val) {
    unsigned int hash = val;
    for (int i = 0; i < 15; ++i) {
        hash = (hash ^ 61) ^ (hash >> 16);
        hash = hash + (hash << 3);
        hash = hash ^ (hash >> 4);
        hash = hash * 0x27d4eb2d;
        hash = hash ^ (hash >> 15);
    }
    return hash;
}

__global__ void gpu_brute_force_kernel(
    unsigned int target_hash,
    unsigned int max_val,
    unsigned int* d_found_src,
    int* d_found_flag
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= max_val) {
        if (custom_hash(idx) == target_hash) {
            if (atomicCAS(d_found_flag, 0, 1) == 0) {
                *d_found_src = idx;
            }
        }
    }
}
""")

# 获取 kernel 函数
kernel = mod.get_function("gpu_brute_force_kernel")

def pycuda_bruteforce(target_hash, max_val_to_check, threads_per_block=256):
    # 主机端输出缓冲
    h_found_src  = np.array([-1], dtype=np.uint32)
    h_found_flag = np.array([0],  dtype=np.int32)

    # 分配设备内存
    d_found_src  = cuda.mem_alloc(h_found_src.nbytes)
    d_found_flag = cuda.mem_alloc(h_found_flag.nbytes)

    # 初始化设备内存
    cuda.memcpy_htod(d_found_src,  h_found_src)
    cuda.memcpy_htod(d_found_flag, h_found_flag)

    # 计算 grid/gridDim
    blocks_per_grid = (int(max_val_to_check) + threads_per_block) // threads_per_block

    # 创建 CUDA 事件用于计时
    start = cuda.Event()
    end   = cuda.Event()

    # 记录开始
    start.record()
    # Launch kernel
    kernel(
        np.uint32(target_hash),
        np.uint32(max_val_to_check),
        d_found_src,
        d_found_flag,
        block=(threads_per_block, 1, 1),
        grid=(blocks_per_grid, 1, 1)
    )
    # 记录结束 & 等待完成
    end.record()
    end.synchronize()

    # 计算耗时
    elapsed_ms = start.time_till(end)

    # 拷贝回主机
    cuda.memcpy_dtoh(h_found_src,  d_found_src)
    cuda.memcpy_dtoh(h_found_flag, d_found_flag)

    return int(h_found_flag[0]), int(h_found_src[0]), elapsed_ms

if __name__ == "__main__":
    # 测试配置（请根据显存/时间酌情调整 max_val_to_check）
    MAX_VALUE_TO_CHECK = 2_000_000_000  # 2e8 以内测试即可
    SOURCE_TO_FIND     = 142385552

    def custom_hash_python(val: int) -> int:
        h = val & 0xFFFFFFFF
        for _ in range(15):
            # step 1: (h ^ 61) ^ (h >> 16)
            h = ((h ^ 61) ^ (h >> 16)) & 0xFFFFFFFF
            # step 2: h + (h << 3)
            h = (h + (h << 3)) & 0xFFFFFFFF
            # step 3: h ^ (h >> 4)
            h = (h ^ (h >> 4)) & 0xFFFFFFFF
            # step 4: h * 0x27d4eb2d
            h = (h * 0x27d4eb2d) & 0xFFFFFFFF
            # step 5: h ^ (h >> 15)
            h = (h ^ (h >> 15)) & 0xFFFFFFFF
        return h


    target_hash = custom_hash_python(SOURCE_TO_FIND)
    print(f"Target hash: {target_hash}, searching for source = {SOURCE_TO_FIND}")
    print(f"Launching PyCUDA kernel on 0…{MAX_VALUE_TO_CHECK}...")

    flag, src, ms = pycuda_bruteforce(target_hash, MAX_VALUE_TO_CHECK)
    if flag:
        print(f"✅ Found source = {src}")
    else:
        print("❌ Not found within range")
    print(f"PyCUDA GPU kernel time: {ms:.3f} ms")
