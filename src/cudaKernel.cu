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
 * CUDA Kernel Device code   ----  vectorAdd
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
 

 /**
 * CUDA Kernel Device code   ----  matrix multiply   Only for Square
 **/
 
 __global__ void
 matrixMul(float *M, float *N, float *P, int width)
 {
    int col= blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < width && col < width)
    {
        float pValue = 0;
        for(int k=0; k<width; k++)
            pValue += M[row * width + k] * N[k * width + col];
        P[row * width + col] = pValue;    
    }
}
 
 /**
 * CUDA Kernel Device code   ----  matrix multiply using shared memory   For rectangle
 **/
 

#define TILE_WIDTH 8

 __global__ void
 matrixMul_sharedMemory(float *M, float *N, float *P, int m, int j, int n)
 {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float pValue = 0;
    for(int ph =0; ph < ceil(j/(float)TILE_WIDTH); ph++)
    {
        if(Row < m && ph * TILE_WIDTH + tx < j)
            Mds[ty][tx] = M[Row * j + ph * TILE_WIDTH + tx];   // M[Row][ph * TILE_WIDTH + tx]
        if(Col < n && ph * TILE_WIDTH + ty < j)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) *n + Col];   // N[ph * TILE_WIDTH + ty][Col]
        __syncthreads();
        
        for(int k = 0; k <TILE_WIDTH; k++)
        {  
            if(ph * TILE_WIDTH + k < j)
                pValue += Mds[ty][k] * Nds[k][tx];
        }
            
        __syncthreads();
    }  
    if(Row < m && Col < n)
        P[Row * n + Col] = pValue;             //  整个代码怎么理解呢？ 要有block并行的想法，每个block都有shared memory，
                                               //  这儿我理解是每个block都申请了Tile_width*Tile_width的内存,以block为单位来想这个程序。 
}
 