#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 

/**
 * Function that multiplies a matrix by a vector
 * (usata anche per il caso per colonne)
 */
void mat_vect_prod(double result_vector[], double *matrix, int rows, int cols, double prod_vector[]) {
    for (int i = 0; i < rows; i++) {
        result_vector[i] = 0;
        for (int j = 0; j < cols; j++) { 
            result_vector[i] += matrix[i * cols + j] * prod_vector[j];
        } 
    }    
}

int main(int argc, char **argv) {
    int i, j;
    int nproc, rank;            
    int rows, cols;                  
    int local_cols;           
    double *matrix = NULL, *local_matrix, *local_result_vector, *result_vector = NULL, *prod_vector = NULL, *local_prod_vector;
    double start, end, mean;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Root legge input e inizializza la matrice e il vettore
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
            printf("\n\nmatrix:\n");
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++)
                    printf("%.3f  ", matrix[i * cols + j]);
                printf("\n");
            }
            fflush(stdout);
        }
    }

    // Broadcast dimensioni
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcolo delle colonne locali
    int base_cols = cols / nproc;
    int remainder = cols % nproc;
    local_cols = base_cols + (rank < remainder ? 1 : 0);
    int local_matrix_size = rows * local_cols;

    // Allocazioni locali
    local_matrix = malloc(local_matrix_size * sizeof(double));
    local_result_vector = malloc(rows * sizeof(double));
    local_prod_vector = malloc(local_cols * sizeof(double));

    // Root prepara sendcounts e displs per la distribuzione per colonne
    int *sendcounts = NULL;
    int *displs = NULL;
    int *vec_sendcounts = NULL;
    int *vec_displs = NULL;

    if (rank == 0) {
        sendcounts = malloc(nproc * sizeof(int));
        displs = malloc(nproc * sizeof(int));
        vec_sendcounts = malloc(nproc * sizeof(int));
        vec_displs = malloc(nproc * sizeof(int));

        displs[0] = 0;
        vec_displs[0] = 0;

        for (i = 0; i < nproc; i++) {
            int local_cols_i = base_cols + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * local_cols_i;
            vec_sendcounts[i] = local_cols_i;

            if (i > 0) {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
                vec_displs[i] = vec_displs[i - 1] + vec_sendcounts[i - 1];
            }
        }
    }

    // Scatterv per parti del vettore
    MPI_Scatterv(
        prod_vector, vec_sendcounts, vec_displs, MPI_DOUBLE,
        local_prod_vector, local_cols, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Scatterv per colonne della matrice
    MPI_Scatterv(
        matrix, sendcounts, displs, MPI_DOUBLE,
        local_matrix, local_matrix_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Stampa locale per debug
    if (cols < 11 && rows < 11) {
        printf("\nRank %d local matrix:\n", rank);
        for (i = 0; i < rows; i++) {
            for (j = 0; j < local_cols; j++)
                printf("%.3f\t", local_matrix[i * local_cols + j]);
            printf("\n");
        }
        printf("local vector (rank %d): ", rank);
        for (j = 0; j < local_cols; j++)
            printf("%.2f ", local_prod_vector[j]);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    // Ogni processo calcola il prodotto parziale
    mat_vect_prod(local_result_vector, local_matrix, rows, local_cols, local_prod_vector);

    // Somma i risultati parziali (somma per righe)
    if (rank == 0)
        result_vector = malloc(rows * sizeof(double));

    MPI_Reduce(local_result_vector, result_vector, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (cols < 11 && rows < 11) {
            printf("\nResult vector:\n");
            for (i = 0; i < rows; i++)
                printf("%f\n", result_vector[i]);
        }

        printf("\nProcs:%d, Time:%f, Matrix:%dx%d\n", nproc, mean / nproc, rows, cols);

        FILE *f = fopen("result_vector_col.txt", "a");
        if (f == NULL) {
            perror("Error file could not be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f, "Procs:%d, Time:%f, Matrix:%dx%d\n", nproc, mean / nproc, rows, cols);
        fclose(f);

        free(matrix);
        free(prod_vector);
        free(result_vector);
        free(sendcounts);
        free(displs);
        free(vec_sendcounts);
        free(vec_displs);
    }

    free(local_matrix);
    free(local_result_vector);
    free(local_prod_vector);

    MPI_Finalize();
    return 0;  
}
