#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "myProto.h"

int main(int argc, char *argv[]) {
	int num_procs, my_rank, size;
	int totalEachProcHistogram[NUMBERS + 1] = { 0 };
	int *totalHistogram;
	int *data = NULL;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	if (num_procs != 2) {
		printf("Run the program with two processes only\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	// Divide the tasks between both processes Master and Slave with MPI
	if (my_rank == MASTER) {
		int lengthData;
		//Init histogram of the project
		totalHistogram = initHistogram(NUMBERS + 1);
		//Read the data from standard input (stdin)
		data = readDataFromStdin(&size);
		lengthData = size;
		size = size / 2;
		if (lengthData % 2 == 0) {

			MPI_Send(&size, 1, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
			MPI_Send(data + size, size, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
		} else // Handle the case that size is odd number and then send lengthdata + 1 to slave.
		{
			lengthData = (lengthData / 2) + 1;
			MPI_Send(&lengthData, 1, MPI_INT, SLAVE, 0, MPI_COMM_WORLD);
			MPI_Send(data + size, lengthData, MPI_INT, SLAVE, 0,
					MPI_COMM_WORLD);
		}
	} else //Slave
	{
		MPI_Recv(&size, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);
		data = (int*) malloc(sizeof(int) * size);

		MPI_Recv(data, size, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD,
				MPI_STATUS_IGNORE);

	}
	// On each process calculate his histogram with OpenMP + Cuda 
	createHistogram(my_rank, data, size, totalEachProcHistogram);

	//Slave send his histogram to Master
	MPI_Reduce(totalEachProcHistogram, totalHistogram, (NUMBERS + 1), MPI_INT,
			MPI_SUM, MASTER, MPI_COMM_WORLD);

	//Master print the program's histogram and free the allocation of total histogram
	if (my_rank == MASTER) {
		printHistogram(totalHistogram, NUMBERS);
		free(totalHistogram);
	}

	free(data);
	MPI_Finalize();
}
