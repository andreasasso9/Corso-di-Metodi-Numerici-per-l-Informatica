#!/bin/bash

# Remove old text files before starting
echo "Removing old .txt files..."
rm -f ./*.txt *.out
echo "All .txt files removed (if any)."
echo

# Number of iterations (how many times to repeat test)
iterations=5
#How many times to double rows/cols
scale_times=5

# MPI process counts for parallel programs
np_values_row=(2 4 8)

echo "===================================="
echo "Starting MPI tests"
echo "===================================="

# 1 Sequential program (fixed np = 1)
echo ">>> Running: seq_sum.out (np = 1)"

# Initial input size
n=16000000

mpicc seq_sum.c -o sum_seq.out 
mpicc parallel_sum.c -o par_sum.out 
mpicc parallel_sum_all.c -o par_sum_all.out 

for ((j=1; j<=scale_times; j++)); do
    for ((i=1; i<=iterations; i++)); do
    echo "--- Iteration $i: n=$n ---"
    mpirun -np 1 --allow-run-as-root ./sum_seq.out "$n" 
    echo "---------------------------------------------"
    done
    # Double input size
    n=$((n * 2))

done


# 2 Parallel program 2nd strategy (variable np)
echo
echo ">>> Running: par_sum.out (np = 2, 4, 8)"

# Initial input size
n=16000000

for ((j=1; j<=scale_times; j++)); do
    for np in "${np_values_row[@]}"; do    
        echo "--- Iteration $i: n=$n ---"
        for ((i=1; i<=iterations; i++)); do
            echo ">> Executing: mpirun -np $np ./par_sum.out  $n"
            mpirun  --oversubscribe --allow-run-as-root -np "$np" ./par_sum.out "$n" 
            echo "---------------------------------------------"
        done
    done
    # Double input size
    n=$((n * 2))
done



# 2 Parallel program 3rd strategy (variable np)
echo
echo ">>> Running: par_sum_all.out (np = 2, 4, 8)"

# Initial input size
n=16000000

for ((j=1; j<=scale_times; j++)); do
    for np in "${np_values_row[@]}"; do    
        echo "--- Iteration $i: n=$n ---"
        for ((i=1; i<=iterations; i++)); do
            echo ">> Executing: mpirun -np $np ./par_sum_all.out   $n"
            mpirun  --oversubscribe --allow-run-as-root -np "$np" ./par_sum_all.out  "$n" 
            echo "---------------------------------------------"
        done
    done
    # Double input size
    n=$((n * 2))
done


echo
echo "All MPI tests completed successfully."
