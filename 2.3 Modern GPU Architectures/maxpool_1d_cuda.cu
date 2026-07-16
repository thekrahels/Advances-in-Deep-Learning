#include <cmath>
#include <torch/extension.h>
#include <ATen/ATen.h>


#define SMART_GROUP 16
#define MAX_BLOCK_SIZE 256
#define LOAD_BLOCK_SIZE 1024


__global__ void maxpool_1d_kernel_brute(int numel, const float* a, int window_size, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float max_val = -INFINITY;

    if (idx < numel) {
        for(int i=0; i<window_size; i++) {
            max_val = max(max_val, a[idx + i]);
        }
        result[idx] = max_val;
    }
}

at::Tensor maxpool_1d_cuda_brute(const at::Tensor& a, int64_t window_size) {
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor result = torch::empty({a_contig.sizes()[0] - window_size + 1}, a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int numel = result.numel();
    maxpool_1d_kernel_brute<<<(numel+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(numel, a_ptr, window_size, result_ptr);
    return result;
}




__global__ void maxpool_1d_kernel_smart(int numel, const float* a, int window_size, float* result) {
    int idx0 = (blockIdx.x * blockDim.x + threadIdx.x) * SMART_GROUP;

#define A(i) ((i) < numel + window_size - 1 ? a[(i)] : -INFINITY)

    // Max value of elements shared within group
    float shared_max_val = -INFINITY;

    for(int i=SMART_GROUP; i<window_size; i++) {
        shared_max_val = max(shared_max_val, A(idx0+i));
    }
    for(int i=0; i<SMART_GROUP; i++) {
        float max_val = shared_max_val;
        for(int j=i; j<SMART_GROUP; j++) {
            max_val = max(max_val, A(idx0+j));
        }
        for(int j=window_size; j<window_size+i; j++) {
            max_val = max(max_val, A(idx0+j));
        }
        result[idx0 + i] = max_val;
    }
}

at::Tensor maxpool_1d_cuda_smart(const at::Tensor& a, int64_t window_size) {
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor result = torch::empty({a_contig.sizes()[0] - window_size + 1}, a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int numel = result.numel();
    int numem_group = (numel + SMART_GROUP - 1) / SMART_GROUP;
    maxpool_1d_kernel_smart<<<(numem_group+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(numel, a_ptr, window_size, result_ptr);
    return result;
}





__global__ void maxpool_1d_kernel_memory(int numel, const float* a, int window_size, float* result) {
    const int bid = blockIdx.x * blockDim.x;

    // Shared between threads in the same block
    __shared__ float a_cache[LOAD_BLOCK_SIZE+MAX_BLOCK_SIZE];

#define A(i) ((i) < numel + window_size - 1 ? a[(i)] : -INFINITY)

    float max_val = -INFINITY;

    for(int j=0; j<window_size; j+=LOAD_BLOCK_SIZE) {
        __syncthreads();
        for (int k = threadIdx.x; k < LOAD_BLOCK_SIZE+MAX_BLOCK_SIZE; k += blockDim.x) {
            a_cache[k] = A(bid+j+k);
        }
        __syncthreads();
        for(int i=j, k=threadIdx.x; i<window_size && i<j+LOAD_BLOCK_SIZE; i++, k++) {
            max_val = max(max_val, a_cache[k]);
        }
    }

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        result[idx] = max_val;
    }
}

at::Tensor maxpool_1d_cuda_memory(const at::Tensor& a, int64_t window_size) {
    TORCH_CHECK(a.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    at::Tensor a_contig = a.contiguous();
    at::Tensor result = torch::empty({a_contig.sizes()[0] - window_size + 1}, a_contig.options());
    const float* a_ptr = a_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    int numel = result.numel();
    maxpool_1d_kernel_memory<<<(numel+MAX_BLOCK_SIZE-1)/MAX_BLOCK_SIZE, MAX_BLOCK_SIZE>>>(numel, a_ptr, window_size, result_ptr);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool_1d_brute", &maxpool_1d_cuda_brute, "Maxpool 1D CUDA (brute force)");
    m.def("maxpool_1d_memory", &maxpool_1d_cuda_memory, "Maxpool 1D CUDA (memory)");
    m.def("maxpool_1d_smart", &maxpool_1d_cuda_smart, "Maxpool 1D CUDA (smart)");
}
