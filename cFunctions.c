#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "myProto.h"

int* readDataFromStdin(int* size){
  int* data;
  int i;
  scanf("%d",size);
  printf("Size is %d\n",*size);
  data = (int*)malloc(sizeof(int)*(*size));
  for(i = 0 ; i < *size ; i++){
     scanf("%d",&data[i]);
  }
  return data;
}

void createHistogram(int rank,int* data, int size, int* totalEachProcHistogram)
{
         if (size%2==0){
          //OpenMP	
          calculateHistogramWithOpenMp(rank ,data, size / 2, totalEachProcHistogram);  
	  //CUDA
          calculateHistogramWithCuda(rank,data + size / 2, size / 2, totalEachProcHistogram);

         }else{
          //OpenMP
          calculateHistogramWithOpenMp(rank ,data, size / 2, totalEachProcHistogram);  
	  //CUDA
          calculateHistogramWithCuda(rank,data + size / 2, (size / 2) +1, totalEachProcHistogram);
         }
}

void calculateHistogramWithCuda(int rank,int* data,int size,int* totalEachProcHistogram) {
	if (HistogramWithCuda(rank,data,size, totalEachProcHistogram) != 0)
		MPI_Abort(MPI_COMM_WORLD, __LINE__);	
}



void calculateHistogramWithOpenMp(int rank,int* data,int size,int* totalEachProcHistogram) {
	
	int i,j,tid,numberOfThreads;
	int* countNumbersPerThread;
	
#pragma omp parallel private(tid) shared(data, countNumbersPerThread, numberOfThreads,size,totalEachProcHistogram)
       {
       	
	tid = omp_get_thread_num();
	numberOfThreads = omp_get_num_threads();

#pragma omp single
	{
		countNumbersPerThread = (int*) malloc(sizeof(int) * ((numberOfThreads * NUMBERS) + 1));
		if (!countNumbersPerThread)
		{
			fprintf(stderr, "Could not allocate array\n");
     	 		MPI_Abort(MPI_COMM_WORLD, 1);	
		}
	}
	
//initiate each histogram of thread with 0
#pragma omp for
	for (int i = 0; i < ((NUMBERS * numberOfThreads)+ 1); i++)
	{
		if(i < NUMBERS){
		   totalEachProcHistogram[i] = 0;
		}
		countNumbersPerThread[i] = 0;
	}
// calcilate  for each thread his own histogram 
#pragma omp for
	for (int i = 0; i < size; i++) 
	{
		countNumbersPerThread[data[i] + tid * NUMBERS]++;
	}
//marge all threads histogram to one global histogram
#pragma omp for
	for (int i = 1; i <= NUMBERS; i++) 
	{
		for (int j = 0; j < numberOfThreads ; j++) 
		{
			totalEachProcHistogram[i] += countNumbersPerThread[i + j * NUMBERS];
		}
	}
    }
    free(countNumbersPerThread);
}

void printHistogram(int* hist, int size) 
{
	int i;
	for (i = 0; i <= size; i++)
	{
		if( hist[i] > 0)
		     printf(" %d:%d\n" ,i, hist[i]);
	}
}

