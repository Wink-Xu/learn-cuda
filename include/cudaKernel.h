
#include <cuda_runtime.h> 


__host__ void 
vectorAdd_host(const float *A, const float *B, float *C, int numElements);

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements);

__global__ void
colorToGray(unsigned char *input, unsigned char *output, int m, int n);

__global__ void
meanFilter(unsigned char *input, unsigned char *output, int m, int n);

__global__ void
matrixMul(float *M, float *N, float *P, int width);

__global__ void
matrixMul_sharedMemory(float *M, float *N, float *P, int m, int k, int j);