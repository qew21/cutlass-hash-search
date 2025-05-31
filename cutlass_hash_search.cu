// File:cutlass_hash_search.cu
// Compile with (adjust CUTLASS_PATH accordingly):
// nvcc -std=c++17 -I/root/cutlass/include -lcuda -lcudart -o cutlass_hash_search cutlass_hash_search.cu

#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

// Include CUTLASS headers (for demonstration; not actively used in the hash logic)
#include <cutlass/cutlass.h>
#include <cutlass/platform/platform.h>

// ---------------------------------------------------------------------------------
// Device‐side “custom_hash” matches the logic in CUTLASS‐inclined examples (15 loops,
// with 32-bit wrap at every step).  This is the same hash used in Triton/PyCUDA/CuPy demos.
// ---------------------------------------------------------------------------------
__device__ __host__ inline unsigned int custom_hash(unsigned int val) {
    unsigned int h = val;
    for (int i = 0; i < 15; ++i) {
        // step 1: (h ^ 61) ^ (h >> 16)
        h = ((h ^ 61) ^ (h >> 16)) & 0xFFFFFFFFu;
        // step 2: h + (h << 3)
        h = (h + (h << 3)) & 0xFFFFFFFFu;
        // step 3: h ^ (h >> 4)
        h = (h ^ (h >> 4)) & 0xFFFFFFFFu;
        // step 4: h * 0x27d4eb2d
        h = (h * 0x27d4eb2d) & 0xFFFFFFFFu;
        // step 5: h ^ (h >> 15)
        h = (h ^ (h >> 15)) & 0xFFFFFFFFu;
    }
    return h;
}

// ---------------------------------------------------------------------------------
// GPU kernel: Brute‐force search.  Each thread processes one “idx” in [0, max_val].
// The first thread to see a matching hash does an atomicCAS on found_flag and writes
// its idx into found_src.
// ---------------------------------------------------------------------------------
__global__ void gpu_brute_force_kernel(
    unsigned int  target_hash,
    unsigned int  max_val_to_check,
    unsigned int* d_found_src,
    int*          d_found_flag
) {
    unsigned int idx = static_cast<unsigned int>(blockIdx.x) * blockDim.x
                       + static_cast<unsigned int>(threadIdx.x);

    if (idx <= max_val_to_check) {
        unsigned int h = custom_hash(idx);
        if (h == target_hash) {
            // Attempt to set the flag from 0 → 1.  Only the first thread succeeds.
            if (atomicCAS(d_found_flag, 0, 1) == 0) {
                *d_found_src = idx;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Host‐side: same hash logic, with explicit 32‐bit masking after each step.
// ---------------------------------------------------------------------------------
unsigned int custom_hash_host(unsigned int val) {
    unsigned int h = val & 0xFFFFFFFFu;
    for (int i = 0; i < 15; ++i) {
        h = ((h ^ 61u) ^ (h >> 16)) & 0xFFFFFFFFu;
        h = (h + (h << 3)) & 0xFFFFFFFFu;
        h = (h ^ (h >> 4)) & 0xFFFFFFFFu;
        h = (h * 0x27d4eb2d) & 0xFFFFFFFFu;
        h = (h ^ (h >> 15)) & 0xFFFFFFFFu;
    }
    return h;
}

// ---------------------------------------------------------------------------------
// Helper: check for CUDA errors after API calls
// ---------------------------------------------------------------------------------
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

// ---------------------------------------------------------------------------------
// MAIN: configure search, pick a random SOURCE, compute its hash on CPU, then
// launch GPU kernel and measure its time with CUDA events.
// ---------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1) Configuration: adjust as needed
    const unsigned int MAX_VALUE_TO_CHECK = 2000'000'000; // e.g., 2e8 for a quick test
    unsigned int SOURCE_TO_FIND;

    // 2) Choose a random source for demonstration (in first 10% of range)
    {
        std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<unsigned long long> dist(0, MAX_VALUE_TO_CHECK / 10ULL);
        SOURCE_TO_FIND = static_cast<unsigned int>(dist(rng));
    }
    // Or uncomment to fix it:
    SOURCE_TO_FIND = 142'385'552u;

    // 3) Compute target hash on CPU
    unsigned int target_hash = custom_hash_host(SOURCE_TO_FIND);
    std::cout << "Target hash: 0x" << std::hex << std::setw(8)
              << std::setfill('0') << target_hash
              << std::dec << ", searching for SOURCE = "
              << SOURCE_TO_FIND << std::endl;
    std::cout << "Range: [0 .. " << MAX_VALUE_TO_CHECK << "]\n\n";

    // 4) Allocate device memory for found_src and found_flag
    unsigned int h_found_src   = 0xFFFFFFFFu; // default “not found”
    int          h_found_flag  = 0;
    unsigned int* d_found_src  = nullptr;
    int*          d_found_flag = nullptr;

    CUDA_CHECK(cudaMalloc(&d_found_src,  sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_found_flag, sizeof(int)));

    // Initialize them on the device
    CUDA_CHECK(cudaMemcpy(d_found_src,  &h_found_src,
                          sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_found_flag, &h_found_flag,
                          sizeof(int),          cudaMemcpyHostToDevice));

    // 5) Decide thread/block layout
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS_PER_GRID  = static_cast<int>((MAX_VALUE_TO_CHECK + THREADS_PER_BLOCK) / THREADS_PER_BLOCK);

    // 6) Create CUDA events for timing
    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));

    // 7) Launch GPU kernel and measure elapsed time
    CUDA_CHECK(cudaEventRecord(start_event));
    gpu_brute_force_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        target_hash,
        MAX_VALUE_TO_CHECK,
        d_found_src,
        d_found_flag
    );
    CUDA_CHECK(cudaEventRecord(end_event));
    CUDA_CHECK(cudaEventSynchronize(end_event));

    float gpu_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start_event, end_event));

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // 8) Copy results back to host
    CUDA_CHECK(cudaMemcpy(&h_found_src,  d_found_src,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int),          cudaMemcpyDeviceToHost));

    // 9) Print outcome
    if (h_found_flag == 1) {
        std::cout << "✅ GPU Found SOURCE = " << h_found_src
                  << " (hash check: 0x" << std::hex << std::setw(8)
                  << custom_hash_host(h_found_src) << std::dec << ")\n";
    } else {
        std::cout << "❌ GPU did NOT find any match in range.\n";
    }
    std::cout << "GPU kernel time: " << std::fixed << std::setprecision(3)
              << gpu_time_ms << " ms\n\n";

    // 10) Clean up
    CUDA_CHECK(cudaFree(d_found_src));
    CUDA_CHECK(cudaFree(d_found_flag));
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));

    return 0;
}
