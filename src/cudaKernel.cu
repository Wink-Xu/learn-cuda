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

         unsigned char r = input[rgbIndex];
         unsigned char g = input[rgbIndex + 1];
         unsigned char b = input[rgbIndex + 2];
         output[pixelIndex] = r* scale[0] + g * scale[1] + b*scale[2];
        //  float res = 0;
        //  for(int i=0; i<3; i++)
        //  {
        //       res += input[rgbIndex+i] * scale[i];
        //  }
         //output[pixelIndex] = res;
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
 

