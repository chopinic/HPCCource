#!/bin/bash

# Request resources:
#SBATCH -N 1		# number of compute nodes. 
#SBATCH -c 101		# number of CPU cores, one per thread, up to 128
#SBATCH --mem=30G	# memory required, up to 250G on standard nodes
#SBATCH --time=0:15:0	# time limit for job (format:  days-hours:minutes:seconds)
#SBATCH -o outPart2Sbatch.out
#SBATCH -p test 

# Run in the 'shared' queue (job may share node with other jobs)

# Modules necessary for job:
module purge
module load gcc


gcc -O3 ./serial.c -o serial
echo "serial output"
time ./serial -NN 512
# compile part2.c into part2
gcc -O3 -fopenmp -lm part2.c -o part2
for threads in 2 4 8 16 32 64 80 100

do
    echo "OMP_NUM_THREADS=$threads"
    time OMP_NUM_THREADS=$threads ./part2 -NN 512
    # Execute your OpenMP program here with the specified number of threads
done
# run part2 with 4 threads
