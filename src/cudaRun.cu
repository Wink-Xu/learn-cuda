#include <stdio.h>
#include <cuda_runtime.h> 
#include <time.h>
#include <cudaKernel.h>
#include <math.h>
#include <utils.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

// *************************
//    1. run_vectorAdd()
//    2. run_colorToGray()
//    3. run_meanFilter()
//    4. run_matrixMul()
// *************************


// *************************
//    Vector Add
// *************************
int run_vectorAdd()
{
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

// *************************
//    color to gray
// *************************

int run_colorToGray()
{
    cv::Mat image = cv::imread("./data/obama.jpg");
    if (image.empty())
    {
        printf(" Wrong Image !!!");
        return 0;
    }else
    {
        printf("Read Image succeed\n Width = %d, Height = %d\n", image.cols, image.rows);
    }

    int width = image.cols;
    int height = image.rows;
    int n_channel = 3;    

    unsigned char *in_data = (unsigned char*)malloc( height * width * 3 * sizeof(unsigned char));
    memset(in_data, 0, height * width * 3 * sizeof(unsigned char));
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            for(int c=0; c<n_channel; c++)
            {
                int _c = 2 - c;
                in_data[i*width*n_channel + j*n_channel + _c] = image.at<Vec3b>(i,j)[c];   //  容易错
            }
        }
    }

    unsigned char *out_data = (unsigned char*)malloc(height * width * sizeof(unsigned char));

    cudaError_t err = cudaSuccess;

    unsigned char *d_in_data = NULL;
    err = cudaMalloc((void **)&d_in_data,  height * width * 3 * sizeof(unsigned char));
    if (err != cudaSuccess)                              //  后面也需要异常判断 ， 先不加
    {
        fprintf(stderr, "Failed to allocate  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    unsigned char *d_out_data =NULL;
    err = cudaMalloc((void **)&d_out_data,  height * width  * sizeof(unsigned char));
    err = cudaMemcpy(d_in_data, in_data, height * width * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_out_data, out_data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);

  // int threadsPerBlock = 1024;   
   dim3 dimGrid(ceil(width/16.0),ceil(height/16.0), 1);                   // 容易犯错   最多可以launch多少个thread？
   dim3 dimBlock(16,16,1);                   // CUDA kernel launch with 10x6 blocks of 16x16 threads
   printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

   colorToGray<<<dimGrid, dimBlock>>>(d_in_data, d_out_data, height, width);

   printf("Copy output data from the CUDA device to the host memory\n");
   err = cudaMemcpy(out_data, d_out_data, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
   if (err != cudaSuccess)                              //  后面也需要异常判断 ， 先不加
   {
       fprintf(stderr, "Failed to allocate  (error code %s)!\n", cudaGetErrorString(err));
       exit(EXIT_FAILURE);
   }


    cv::Mat grayImg = Array2Mat(out_data, height, width);
    cv::imwrite("./obama_gray.jpg", grayImg);
   
   printf("Test PASSED\n");

   // Free device global memory
   err = cudaFree(d_in_data);
   err = cudaFree(d_out_data);

   // Free host memory
   free(in_data);
   free(out_data);

   printf("Done\n");

    return 0;
}

// *************************
//    mean filter
// *************************

int run_meanFilter()
{
    cv::Mat image = cv::imread("./data/obama_gray.jpg", CV_8U);
    if (image.empty())
    {
        printf(" Wrong Image !!!");
        return 0;
    }else
    {
        printf("Read Image succeed\n Width = %d, Height = %d\n", image.cols, image.rows);
    }

    int width = image.cols;
    int height = image.rows; 

    unsigned char *in_data = (unsigned char*)malloc( height * width * sizeof(unsigned char));
    memset(in_data, 0, height * width * sizeof(unsigned char));
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            in_data[i*width + j] = image.at<unsigned char>(i,j);  
        }
    }

    unsigned char *out_data = (unsigned char*)malloc(height * width * sizeof(unsigned char));

    cudaError_t err = cudaSuccess;

    unsigned char *d_in_data = NULL;
    err = cudaMalloc((void **)&d_in_data,  height * width * sizeof(unsigned char));
    if (err != cudaSuccess)                              //  后面也需要异常判断 ， 先不加
    {
        fprintf(stderr, "Failed to allocate  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    unsigned char *d_out_data =NULL;
    err = cudaMalloc((void **)&d_out_data,  height * width  * sizeof(unsigned char));
    err = cudaMemcpy(d_in_data, in_data, height * width *  sizeof(unsigned char), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_out_data, out_data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // int threadsPerBlock = 1024;   
    dim3 dimGrid(ceil(width/16.0),ceil(height/16.0), 1);                  
    dim3 dimBlock(16,16,1);                  
    printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    meanFilter<<<dimGrid, dimBlock>>>(d_in_data, d_out_data, height, width);

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(out_data, d_out_data, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)                             
    {
        fprintf(stderr, "Failed to allocate  (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    cv::Mat grayImg = Array2Mat(out_data, height, width);
    cv::imwrite("./obama_gray_blur.jpg", grayImg);
    printf("Test PASSED\n");

   // Free device global memory
    err = cudaFree(d_in_data);
    err = cudaFree(d_out_data);

    // Free host memory
    free(in_data);
    free(out_data);

    printf("Done\n");

    return 0;
}

// *************************
//    Matrix Multiply
// *************************
int run_matrixMul()
{

    // Print the vector length to be used, and compute its size
    int mRow = 256;
    int mCol = 128;
    int nRow = 128;
    int nCol = 64;
    size_t sizeM = mRow * mCol * sizeof(float);
    size_t sizeN = nRow * nCol * sizeof(float);
    printf("[Matrix multiply of %dx%d * %dx%d]\n", mRow, mCol, nRow, nCol);

    // Allocate the host vector 
    float *h_M = (float *)malloc(sizeM);
    float *h_N = (float *)malloc(sizeN);
    float *h_P = (float *)malloc(mRow * nCol * sizeof(float));

    // Initialize the host input vectors
    for (int i = 0; i < mRow * mCol; ++i)
    {
        h_M[i] = 1;
    }
    for (int i = 0; i < nRow * nCol; ++i)
    {
        h_N[i] = 1;
    }    
  
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    // Allocate the device vector
    float *d_M = NULL;
    err = cudaMalloc((void **)&d_M, sizeM);

    if (err != cudaSuccess)                              //  后面也需要异常判断 ， 先不加
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_N = NULL;
    err = cudaMalloc((void **)&d_N, sizeN);
    float *d_P = NULL;
    err = cudaMalloc((void **)&d_P, mRow * nCol * sizeof(float));
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(mRow/16.0),ceil(nCol/16.0), 1);                  
    dim3 dimBlock(16,16,1);                  
    printf("CUDA kernel launch with %dx%d blocks of %dx%d threads\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);

    matrixMul<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, mCol);

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_P, d_P, mRow * nCol * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f\t %f\n", h_P[0], h_P[1]);

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_M);
    err = cudaFree(d_N);
    err = cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    printf("Done\n");
    return 0;

}