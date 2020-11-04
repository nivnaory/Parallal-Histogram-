#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "myProto.h"

__global__ void HistogramKernel(const int* a , int* dev_globalHist, int size)
{
    //Each block has a private histogram that can be accessed through the shared memory between all the threads in the block.
    __shared__ int sharedHistogram[NUMBERS + 1];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = threadIdx.x + 1;
    sharedHistogram[threadIdx.x] = 0;
    if(i < size){
    	atomicAdd(&sharedHistogram[a[i]], 1);
    }
    __syncthreads();
    atomicAdd(&dev_globalHist[j],sharedHistogram[j]);
}

int HistogramWithCuda(int rank,int* arr, int size, int* totalEachProcHistogram)
{
    int* dev_arr = 0;
    int globalHist[NUMBERS + 1] = {0};
    int *dev_globalHist;
    cudaError_t cudaStatus;   
  

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    // Allocate GPU buffers for data (input).
    cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }

    // Allocate GPU buffers for gloabl histogram.
    cudaStatus = cudaMalloc((void**)&dev_globalHist, (NUMBERS+1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return 1;
    }


    // Copy input data from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

    cudaStatus = cudaMemcpy(dev_globalHist,globalHist ,(NUMBERS +1) * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            return 1;
        }

    // Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(size + threadsPerBlock - 1) / threadsPerBlock;
    HistogramKernel <<<blocksPerGrid, threadsPerBlock >>> (dev_arr,dev_globalHist , size);

    // Copy output global histogram from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(globalHist, dev_globalHist, (NUMBERS+1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return 1;
    }

     for (int i=0;i<=NUMBERS;i++)
     {
       totalEachProcHistogram[i]+=globalHist[i];
     }
     
    cudaFree(dev_arr);
    cudaFree(dev_globalHist);

    return 0;
}

