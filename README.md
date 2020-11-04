# Parallal-Histogram-
a project, that calculate histogram of given random array (read from file)  by using parallel programing , The calculation is done by Cuda and OpenMp . 
the project take a an array split the array by 2. 
by using Mpi we  split the jobs to 2 different proccess.  
then, each proccess split the half array  by 2 again and split the jobs  one to opneMp and one  to cuda. 
in the end, we merge the result into 1 complete array(root process take care of this by using Mpi).
