# notebook

## chapter 3

# 1.CUDA Thread Organization 

```c
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(…);
```

maxThreadsPerBlock : 1024.           
maxThreadsDim[0 - 2] : 1024 1024 64. 
maxGridSize[0 - 2] : 2147483647 65535 65535.  


```C
int col= blockDim.x * blockIdx.x + threadIdx.x;
int row = blockDim.y * blockIdx.y + threadIdx.y;
if (row < height && col < width)
{
    int pixelIndex = row * width + col;
}
```

col = x = width;  row = y = height

# 2.code example: 'colorToGray' , 'meanFilter'
```C
/**
 * CUDA Kernel Device code   ----  colorToGray
 **/
 __global__ void
 colorToGray(unsigned char *input, unsigned char *output, int height, int width)
 {
     int col= blockDim.x * blockIdx.x + threadIdx.x;
     int row = blockDim.y * blockIdx.y + threadIdx.y;
     float scale[3] = {0.299, 0.587, 0.114};
     if (row < height && col < width)
     {
         int pixelIndex = row * width + col;
         int rgbIndex = pixelIndex * 3;

         unsigned char r = input[rgbIndex];                 // rgb rgb rgb rgb rgb
         unsigned char g = input[rgbIndex + 1];
         unsigned char b = input[rgbIndex + 2];
         output[pixelIndex] = r* scale[0] + g * scale[1] + b*scale[2];
     }
 }

 /**
 * CUDA Kernel Device code   ----  mean filter
 **/
#define FILTER_SIZE 3

 __global__ void
 meanFilter(unsigned char *input, unsigned char *output, int height, int width)
 {
    int col= blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < height && col < width)
    {
        int pixelIndex = row * width + col;
        int pixelNum = 0;
        int tempSum = 0;
        for(int i = -FILTER_SIZE + 1; i <  FILTER_SIZE; i++)
        {
            for(int j = -FILTER_SIZE + 1; j < FILTER_SIZE; j++ )
            {
                if(col + i >= 0 && col + i < width && row + j >= 0 && row + j < height)
                {
                    tempSum += input[(row + j) * width + col +i];
                    pixelNum++;
                } 
            }
        }
        output[pixelIndex] = tempSum/pixelNum;
    }
}
```

# 3.SYNCHRONIZATION AND TRANSPARENT SCALABILITY
```
__syncthreads()
```
* When the latest one in the block arrives at the barrier, everyone can continue their execution. With barrier synchronization, “No one is left behind.”

* When a __syncthread() statement is placed in an if-statement, either all or none of the threads in a block execute the path that includes the __syncthreads()

* one needs to make sure that all threads involved in the barrier synchronization have access to the necessary resources to eventually arrive at the barrier.(When a thread of a block is assigned to an execution resource, all other threads in the same block are also assigned to the same resource)

* TRANSPARENT SCALABILITY
  threads in different blocks cannot synchronize with one another

# 4.RESOURCE ASSIGNMENT

* Threads are assigned to SMs for execution on a block-by-block basis

* Each device sets a limit on the number of blocks that can be assigned to each SM.(If a CUDA device has 30 SMs, and each SM can accommodate up to 1536 threads, the device can have up to 46,080 threads simultaneously residing in the CUDA device for execution)

* In situations where there is shortage of one or more types of resources needed for the simultaneous execution of 8 blocks, The runtime system maintains a list of blocks that need to execute and assigns new blocks to SMs as previously assigned blocks complete execution.

# 5.THREAD SCHEDULING AND LATENCY TOLERANCE

* a block assigned to an SM is further divided into 32 thread units
called warps, Warps are not part of the CUDA specification, the size of warps is a property of a CUDA device.(SMs-->blocks-->warps-->32thread)

* An SM is designed to execute all threads in a warp following the Single Instruction, Multiple Data (SIMD) model—i.e., at any instant in time, one instruction is fetched and executed for all threads in the warp.

* In general, there are fewer SPs than the threads assigned to each SM; i.e., each SM has only enough hardware to execute instructions from a small subset of all threads assigned to the SM at any point in time.

* At any time, the SM executes instructions of only a small subset of its resident warps. This condition allows the other warps to wait for long-latency operations without slowing down the overall execution throughput of the massive number of execution units



