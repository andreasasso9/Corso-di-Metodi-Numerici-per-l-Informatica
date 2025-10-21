#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

/**
 * Function that multiplies a matrix by a vector
 */
void mat_vect_prod(double result_vector[], double *matrix, int rows, int cols, double prod_vector[]) {
    for (int i = 0; i < rows; i++) {
        result_vector[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result_vector[i] += matrix[i * cols + j] * prod_vector[j];
        }
    }
}

int main(int argc, char **argv) {

    int i, j;
    int nproc, rank;            
    int rows, cols;              
    int local_rows;              
    double *matrix = NULL, *local_matrix, *local_result_vector, *result_vector = NULL, *prod_vector;
    double start, end, mean;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Root process: initialize matrix and vector
    if (rank == 0) {
        if (argc > 3) {
            rows = atoi(argv[2]);
            cols = atoi(argv[3]);
        } else {
            printf("Insert rows number = ");
            fflush(stdout);
            scanf("%d", &rows);

            printf("Insert columns number = ");
            fflush(stdout);
            scanf("%d", &cols);
        }

        matrix = malloc(rows * cols * sizeof(double));
        prod_vector = malloc(cols * sizeof(double));
        result_vector = malloc(rows * sizeof(double));

        for (j = 0; j < cols; j++)
            prod_vector[j] = j;

        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                if (j == 0)
                    matrix[i * cols + j] = 1.0 / (i + 1) - 1;
                else
                    matrix[i * cols + j] = 1.0 / (i + 1) - pow(1.0 / 2.0, j);
            }
        }

        if (cols < 11 && rows < 11) {
            printf("\nv: \n");
            for (j = 0; j < cols; j++)
                printf("%.2f ", prod_vector[j]);
            printf("\n\nmatrix: \n");
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++)
                    printf("%.3f  ", matrix[i * cols + j]);
                printf("\n");
            }
            fflush(stdout);
        }
    }

    // Broadcast matrix dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast the product vector to all processes
    if (rank != 0)
        prod_vector = malloc(cols * sizeof(double));
    MPI_Bcast(prod_vector, cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute local rows for each process
    int base_rows = rows / nproc;
    int remainder = rows % nproc;
    local_rows = base_rows + (rank < remainder ? 1 : 0);
    int local_matrix_size = local_rows * cols;

    local_matrix = malloc(local_matrix_size * sizeof(double));
    local_result_vector = malloc(local_rows * sizeof(double));

    // Root prepares sendcounts and displacements
    int *sendcounts = NULL, *displs = NULL;
    int *recvcounts = NULL, *recvdispls = NULL;

    if (rank == 0) {
        sendcounts = malloc(nproc * sizeof(int));
        displs = malloc(nproc * sizeof(int));
        recvcounts = malloc(nproc * sizeof(int));
        recvdispls = malloc(nproc * sizeof(int));

        displs[0] = 0;
        recvdispls[0] = 0;

        for (i = 0; i < nproc; i++) {
            int local_rows_i = base_rows + (i < remainder ? 1 : 0);
            sendcounts[i] = local_rows_i * cols;
            recvcounts[i] = local_rows_i;

            if (i > 0) {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
                recvdispls[i] = recvdispls[i - 1] + recvcounts[i - 1];
            }
        }
    }

    // Scatter rows of the matrix to each process
    MPI_Scatterv(
        matrix, sendcounts, displs, MPI_DOUBLE,
        local_matrix, local_matrix_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Print local matrices if small
    if (cols < 11 && local_rows < 11) {
        printf("Rank %d local matrix:\n", rank);
        for (i = 0; i < local_rows; i++) {
            for (j = 0; j < cols; j++)
                printf("%.3f  ", local_matrix[i * cols + j]);
            printf("\n");
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Local computation
    mat_vect_prod(local_result_vector, local_matrix, local_rows, cols, prod_vector);

    // Gather results
    MPI_Gatherv(
        local_result_vector, local_rows, MPI_DOUBLE,
        result_vector, recvcounts, recvdispls, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (cols < 11 && rows < 11) {
            printf("\nResult vector:\n");
            for (i = 0; i < rows; i++)
                printf("%f\n", result_vector[i]);
        }

        printf("\nProcs: %d, Time: %f, Matrix: %dx%d\n",
               nproc, mean / nproc, rows, cols);

        FILE *f = fopen("result_vector_row.txt", "a");
        if (f == NULL) {
            perror("Error file could not be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f, "Procs:%d, Time:%f, Matrix:%dx%d\n",
                nproc, mean / nproc, rows, cols);
        fclose(f);

        free(matrix);
        free(result_vector);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);
    }

    free(local_matrix);
    free(local_result_vector);
    free(prod_vector);

    MPI_Finalize();
    return 0;
}
