# Parallal-Histogram-
a project, that calculate histogram of given random array (read from file)  by using parallel programing , The calculation is done by Cuda and OpenMp . 
the project take a an array split the array by 2 and by using Mpi, split the job to 2 different proccess.  
then,  each proccess split the helf array  by 2 agin and split the job 1 to opneMp and 1 to cuda. 
in the end, we merge the result into 1 complete array(root process take care of this by using Mpi).
