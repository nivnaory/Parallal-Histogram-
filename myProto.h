#pragma once

#define NUMBERS 256
#define NUMBER_OF_THREADS 4
#define MASTER 0
#define SLAVE 1

int* readDataFromStdin(int *size);
int* initHistogram(int size);
void createHistogram(int rank, int *data, int size, int *totalHistogram);
void calculateHistogramWithOpenMp(int rank, int *data, int size,int *totalHistogram);
void calculateHistogramWithCuda(int rank, int *data, int size,int *totalEachProcHistogram);
int HistogramWithCuda(int rank, int *data, int size,int *totalEachProcHistogram);
void printHistogram(int *hist, int size);

