/**
 * Vector addition: C = A + B.
**/
#include <stdio.h>
//#include <cuda.h>
#include <cuda_runtime.h>   // 怎么知道 需要哪些头文件 需要哪些动态库
#include <time.h>
#include <getDeviceProp.h>

/**
 * CPU Kernel host code
 **/
__host__ void 
vectorAdd_host(const float *A, const float *B, float *C, int numElements)
{
    for(int i =0; i<numElements; i++)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * CUDA Kernel Device code
 **/
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
int main(void)
{
    getDeviceProp();

    printf("********************************************\n");


    clock_t time1,time2,time3,time4,time5;

    // Print the vector length to be used, and compute its size
    int numElements = 500000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host vector 
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1;
        h_B[i] = 1;
    }
    
    // Run the host program
    time1 = clock();
    vectorAdd_host(h_A, h_B, h_C, numElements); 
    time2 = clock();

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // Allocate the device vector
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)                              //  后面也需要异常判断 ， 先不加
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;                      //  这个每个block里的threads是怎么设定的？  书上写的是256 我设置成 10240与50000都可以。 不会报错但是算出来的错了。
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    time4 = clock();
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    time5 = clock();

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("%f\t %f\n", h_C[0], h_C[499999]);

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    time3 = clock();
    printf("Host vectorAdd program running time is %f\n", (double)(time2 - time1)/CLOCKS_PER_SEC);
    printf("Device vectorAdd program time is %f\n", (double)(time5- time4)/CLOCKS_PER_SEC);
    printf("All process of VectorAdd program in GPU running time is %f\n", (double)(time3 - time2)/CLOCKS_PER_SEC);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}





/**

Device Name : GeForce GTX 1080 Ti.   用于标识设备的ASCII字符串;
totalGlobalMem : -1163395072.        线程块可以使用的共享存储器的最大值,以字节为单位;多处理器上的所有线程块可以同时共享这些存储器;
sharedMemPerBlock : 49152.           线程块可以使用的共享存储器的最大值,以字节为单位;多处理器上的所有线程块可以同时共享这些存储器;
regsPerBlock : 65536.                线程块可以使用的32位寄存器的最大值;多处理器上的所有线程块可以同时共享这些寄存器;
warpSize : 32.                       按线程计算的warp块大小;
memPitch : 2147483647.               允许通过cudaMallocPitch()为包含存储器区域的存储器复制函数分配的最大间距(pitch),以字节为单位;
maxThreadsPerBlock : 1024.           每个块中的最大线程数
maxThreadsDim[0 - 2] : 1024 1024 64. 块各个维度的最大值:
maxGridSize[0 - 2] : 2147483647 65535 65535.   网格各个维度的最大值;
totalConstMem : 65536.               设备上可用的不变存储器总量,以字节为单位;
major.minor : 6.1.                   定义设备计算能力的主要修订号和次要修订号;
clockRate : 1582000.                 以千赫为单位的时钟频率;
textureAlignment : 512.              对齐要求;与textureAlignment字节对齐的纹理基址无需对纹理取样应用偏移
deviceOverlap : 1.                   如果设备可在主机和设备之间并发复制存储器,同时又能执行内核,则此值为 1;否则此值为 0;
multiProcessorCount : 28.            设备上多处理器的数量

********************************************
[Vector addition of 500000 elements]
Copy input data from the host memory to the CUDA device
CUDA kernel launch with 489 blocks of 1024 threads
Copy output data from the CUDA device to the host memory
2.000000         2.000000
Test PASSED

Host vectorAdd program running time is 0.004658
Device vectorAdd program time is 0.000025
All process of VectorAdd program in GPU running time is 0.217716

Done




硬件概念与软件概念的对应。    硬件 SM->warp->sp  软件 grid->block->thread 如何对应于整合的？
为啥不用连接库 因为nvcc 就像g++也不需要连接所有的库 
sms？  GENCODE_FLAGS
**/