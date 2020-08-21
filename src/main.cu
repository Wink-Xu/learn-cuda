#include <stdio.h>
#include <time.h>
#include <getDeviceProp.h>
#include <cudaRun.h>


int main(void)
{
    getDeviceProp();
    printf("********************************************\n");
    
//  run_vectorAdd();
//  run_colorToGray();  
    run_meanFilter();
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