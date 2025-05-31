#include <iostream>
#include <vector>
#include <chrono>
#include <random> // For picking a random number to find
#include <iomanip> // For std::fixed, std::setprecision

// --- Custom "Hashing" Function (computationally intensive) ---
// This is not a cryptographically secure hash, just for demonstration of work.
__host__ __device__ unsigned int custom_hash(unsigned int val) {
    unsigned int hash = val;
    for (int i = 0; i < 15; ++i) { // Loop to increase work
        hash = (hash ^ 61) ^ (hash >> 16);
        hash = hash + (hash << 3);
        hash = hash ^ (hash >> 4);
        hash = hash * 0x27d4eb2d; // FNV-1a prime
        hash = hash ^ (hash >> 15);
    }
    return hash;
}

// --- CPU Brute-Force Search ---
long long cpu_brute_force_search(unsigned int target_hash, unsigned int max_val_to_check, unsigned int& found_source) {
    for (unsigned int i = 0; i <= max_val_to_check; ++i) {
        if (custom_hash(i) == target_hash) {
            found_source = i;
            return i; // Return the found value
        }
    }
    found_source = (unsigned int)-1; // Indicate not found
    return -1;
}

// --- CUDA Kernel for Brute-Force Search ---
__global__ void gpu_brute_force_kernel(unsigned int target_hash, unsigned int max_val_to_check, unsigned int* d_found_source, int* d_found_flag) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx <= max_val_to_check) {
        if (custom_hash(idx) == target_hash) {
            // Potential race condition if multiple threads find it,
            // but for one specific target, only one should match.
            // Using atomicCAS to ensure only the first write succeeds to d_found_flag
            if (atomicCAS(d_found_flag, 0, 1) == 0) { // If d_found_flag was 0, set to 1 and proceed
                 *d_found_source = idx;
            }
        }
    }
}

int main() {
    // --- Configuration ---
    unsigned int max_value_to_check = 2 * 1000 * 1000 * 1000; // 20亿 - 调整这个值来改变搜索空间和时间
    unsigned int source_to_find; // The number whose hash we want to find

    // Randomly pick a number within a smaller range to ensure it's findable by CPU in reasonable demo time
    // but still within the larger search space for GPU.
    // For a real demo, you might want to set this manually for consistency.
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> distrib(0, max_value_to_check / 10); // Pick from the first 0.1%
    source_to_find = 142385552; // Example

    unsigned int target_hash_value = custom_hash(source_to_find);

    std::cout << "Target Hash Value: " << target_hash_value << std::endl;
    std::cout << "(Will try to find the source value: " << source_to_find << ")" << std::endl;
    std::cout << "Max Value to Check: " << max_value_to_check << std::endl << std::endl;

    unsigned int cpu_found_source = (unsigned int)-1;
    unsigned int gpu_found_source_host = (unsigned int)-1;


    // --- CPU Brute-Force ---
    std::cout << "Starting CPU brute-force search..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_brute_force_search(target_hash_value, max_value_to_check, cpu_found_source);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration_ms = end_cpu - start_cpu;

    if (cpu_found_source != (unsigned int)-1) {
        std::cout << "CPU Found Source: " << cpu_found_source << " (Hash: " << custom_hash(cpu_found_source) << ")" << std::endl;
    } else {
        std::cout << "CPU: Source not found within range." << std::endl;
    }
    std::cout << "CPU search time: " << std::fixed << std::setprecision(3) << cpu_duration_ms.count() << " ms" << std::endl;


    // --- GPU Brute-Force ---
    std::cout << "\nStarting GPU brute-force search..." << std::endl;
    unsigned int* d_found_source_gpu; // Device memory for the found source
    int* d_found_flag_gpu;         // Device memory for a flag to indicate if found

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_found_source_gpu, sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) { std::cerr << "cudaMalloc d_found_source_gpu failed!" << std::endl; return 1; }
    cudaStatus = cudaMalloc((void**)&d_found_flag_gpu, sizeof(int));
    if (cudaStatus != cudaSuccess) { std::cerr << "cudaMalloc d_found_flag_gpu failed!" << std::endl; return 1; }

    // Initialize d_found_flag to 0 (false) on the device
    int h_found_flag_initial = 0;
    cudaMemcpy(d_found_flag_gpu, &h_found_flag_initial, sizeof(int), cudaMemcpyHostToDevice);
    // Initialize d_found_source_gpu to a known invalid state (optional, but good practice)
    unsigned int h_found_source_initial = (unsigned int)-1;
    cudaMemcpy(d_found_source_gpu, &h_found_source_initial, sizeof(unsigned int), cudaMemcpyHostToDevice);


    // CUDA execution timing
    cudaEvent_t start_gpu_event, stop_gpu_event;
    cudaEventCreate(&start_gpu_event);
    cudaEventCreate(&stop_gpu_event);

    // Kernel launch parameters
    int threads_per_block = 256;
    // Number of blocks needed to cover max_value_to_check + 1 elements
    int blocks_per_grid = (max_value_to_check + 1 + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start_gpu_event);
    gpu_brute_force_kernel<<<blocks_per_grid, threads_per_block>>>(target_hash_value, max_value_to_check, d_found_source_gpu, d_found_flag_gpu);
    cudaEventRecord(stop_gpu_event);
    cudaEventSynchronize(stop_gpu_event); // Wait for kernel to complete

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "GPU kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Fall through to cleanup
    }

    float gpu_kernel_duration_ms = 0;
    cudaEventElapsedTime(&gpu_kernel_duration_ms, start_gpu_event, stop_gpu_event);

    // Copy result back from GPU
    int h_found_flag_result = 0;
    cudaMemcpy(&h_found_flag_result, d_found_flag_gpu, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_found_flag_result == 1) {
        cudaMemcpy(&gpu_found_source_host, d_found_source_gpu, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::cout << "GPU Found Source: " << gpu_found_source_host << " (Hash: " << custom_hash(gpu_found_source_host) << ")" << std::endl;
    } else {
        std::cout << "GPU: Source not found within range." << std::endl;
    }
    std::cout << "GPU kernel search time: " << std::fixed << std::setprecision(3) << gpu_kernel_duration_ms << " ms" << std::endl;

    // --- Results & Speedup ---
    if (cpu_found_source != (unsigned int)-1 && gpu_found_source_host != (unsigned int)-1 && cpu_duration_ms.count() > 0 && gpu_kernel_duration_ms > 0) {
         if (cpu_found_source == gpu_found_source_host) {
            std::cout << "\nVerification: CPU and GPU found the same source value." << std::endl;
         } else {
            std::cout << "\nError: CPU and GPU found different source values or one did not find it." << std::endl;
         }
        double speedup = cpu_duration_ms.count() / gpu_kernel_duration_ms;
        std::cout << "Speedup (CPU Time / GPU Kernel Time): " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    } else if (cpu_duration_ms.count() == 0 || gpu_kernel_duration_ms == 0) {
        std::cout << "\nCannot calculate speedup due to zero execution time (range might be too small or error occurred)." << std::endl;
    }


    // Cleanup
    cudaFree(d_found_source_gpu);
    cudaFree(d_found_flag_gpu);
    cudaEventDestroy(start_gpu_event);
    cudaEventDestroy(stop_gpu_event);

    return 0;
}