#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "myProto.h"

int* readDataFromStdin(int *size) {
	int *data;
	int i;
	scanf("%d", size);
	printf("Size is %d\n", *size);
	data = (int*) malloc(sizeof(int) * (*size));
	for (i = 0; i < *size; i++) {
		scanf("%d", &data[i]);
	}
	return data;
}

int* initHistogram(int size) {
	int i;
	int *totalHistogram = (int*) malloc(sizeof(int) * size);
	for (i = 0; i < size; i++) {
		totalHistogram[i] = 0;
	}
	return totalHistogram;
}

void createHistogram(int rank, int *data, int size, int *totalEachProcHistogram) {
	if (size % 2 == 0) {
		//OpenMP
		calculateHistogramWithOpenMp(rank, data, size / 2,
				totalEachProcHistogram);
		//CUDA
		calculateHistogramWithCuda(rank, data + size / 2, size / 2,
				totalEachProcHistogram);

	} else { // // Handle that case that size / 2 isn't even number such as (14 , 22 )
		//OpenMP
		calculateHistogramWithOpenMp(rank, data, size / 2,
				totalEachProcHistogram);
		//CUDA
		calculateHistogramWithCuda(rank, data + size / 2, (size / 2) + 1,
				totalEachProcHistogram);
	}
}

//Calculate part of histogram on GPU with Cuda
void calculateHistogramWithCuda(int rank, int *data, int size,
		int *totalEachProcHistogram) {
	if (HistogramWithCuda(rank, data, size, totalEachProcHistogram) != 0)
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
}

//Calculate part of histogram on CPU with OpenMP
void calculateHistogramWithOpenMp(int rank, int *data, int size,
		int *totalEachProcHistogram) {

	int i, j, tid, numberOfThreads;
	int *countNumbersPerThread;

#pragma omp parallel private(tid) shared(data, countNumbersPerThread, numberOfThreads, size, totalEachProcHistogram)
	{

		tid = omp_get_thread_num();
		numberOfThreads = omp_get_num_threads();
//One thread will allocate memory for the array that contains within it all the private histograms
#pragma omp single
		{
			countNumbersPerThread = (int*) malloc(
					sizeof(int) * ((numberOfThreads * NUMBERS)));
			if (!countNumbersPerThread) {
				fprintf(stderr, "Could not allocate array\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}

//initiate each histogram of thread with 0
#pragma omp for
		for (int i = 1; i <= ((NUMBERS * numberOfThreads)); i++) {
			if (i <= NUMBERS) {
				totalEachProcHistogram[i] = 0;
			}
			countNumbersPerThread[i] = 0;
		}

//Each thread computes its own private histogram 
#pragma omp for
		for (int i = 0; i < size; i++) {
			if (data[i] != 0) {
				countNumbersPerThread[data[i] + tid * NUMBERS]++;
			}

		}

//Merge the private histograms of each thread into a global histogram
#pragma omp for
		for (int i = 1; i <= NUMBERS; i++) {
			for (int j = 0; j < numberOfThreads; j++) {
				totalEachProcHistogram[i] += countNumbersPerThread[i
						+ j * NUMBERS];
			}
		}
	}
	free(countNumbersPerThread);

}

//Print Histogram
void printHistogram(int *hist, int size) {
	int i;
	for (i = 0; i <= size; i++) {
		if (hist[i] > 0)
			printf(" %d:%d\n", i, hist[i]);
	}
}
