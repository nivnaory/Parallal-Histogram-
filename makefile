build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c cudaHistogram.cu -o cudaHistogram.o
	mpicxx -fopenmp -o mpiCudaOpenMP  main.o cFunctions.o  cudaHistogram.o  /usr/local/cuda-11.1/lib64/libcudart_static.a -ldl -lrt
run:
	mpiexec -np 2 ./mpiCudaOpenMP $(FILE)
clean:
	rm -f *.o ./mpiCudaOpenMP
                                                                                                                                    
