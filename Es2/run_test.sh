#!/bin/bash

# Remove old text files before starting
echo "Removing old .txt files..."
rm -f ./*.txt
echo "All .txt files removed (if any)."
echo

# Number of iterations (how many times to repeat test)
iterations=5
#How many times to double rows/cols
scale_times=5

# MPI process counts for mat-vet-row.out
np_values_row=(2 4 8)

echo "===================================="
echo "Starting MPI tests"
echo "===================================="

# 1 Sequential program (fixed np = 1)
echo ">>> Running: mat-vet-seq.out (np = 1)"

# Initial matrix size
rows=100
cols=120

for ((j=1; j<=scale_times; j++)); do
    for ((i=1; i<=iterations; i++)); do
    echo "--- Iteration $i: rows=$rows, cols=$cols ---"
    mpirun -np 1 ./mat-vet-seq.out . "$rows" "$cols"
    echo "---------------------------------------------"
    done
    # Double matrix size
    rows=$((rows * 2))
    cols=$((cols * 2))
done


# 2 Parallel program with row distribution (variable np)
echo
echo ">>> Running: mat-vet-row.out (np = 2, 4, 8)"

# Initial matrix size
rows=100
cols=120

for ((j=1; j<=scale_times; j++)); do
    for np in "${np_values_row[@]}"; do    
        echo "--- Iteration $i: rows=$rows, cols=$cols ---"
        for ((i=1; i<=iterations; i++)); do
            echo ">> Executing: mpirun -np $np ./mat-vet-row.out . $rows $cols"
            mpirun  --oversubscribe -np "$np" ./mat-vet-row.out . "$rows" "$cols"
            echo "---------------------------------------------"
        done
    done
    # Double matrix size
    rows=$((rows * 2))
    cols=$((cols * 2))
done



# 3 Parallel program with column distribution (variable np)
echo
echo ">>> Running: mat-vet-col.out (np = 2, 4, 8)"

# Initial matrix size
rows=100
cols=120

for ((j=1; j<=scale_times; j++)); do
    for np in "${np_values_row[@]}"; do    
        echo "--- Iteration $i: rows=$rows, cols=$cols ---"
        for ((i=1; i<=iterations; i++)); do
            echo ">> Executing: mpirun -np $np ./mat-vet-col.out . $rows $cols"
            mpirun  --oversubscribe -np "$np" ./mat-vet-col.out . "$rows" "$cols"
            echo "---------------------------------------------"
        done
    done
    # Double matrix size
    rows=$((rows * 2))
    cols=$((cols * 2))
done

echo
echo "All MPI tests completed successfully."
