#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* Funzione originale */
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
    int rows = 0, cols = 0;
    int local_cols;
    double *matrix = NULL;
    double *local_matrix = NULL;
    double *local_result_vector = NULL;
    double *result_vector = NULL;
    double *prod_vector = NULL;
    double *local_prod_vector = NULL;
    double start, end, mean;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Root prepara i dati */
    if (rank == 0) {
        if (argc > 1) {
            rows = atoi(argv[1]);
            cols = atoi(argv[2]);
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

        for (j = 0; j < cols; j++) prod_vector[j] = j;

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
            for (j = 0; j < cols; j++) printf("%.2f ", prod_vector[j]);
            printf("\n\nmatrix:\n");
            for (i = 0; i < rows; i++) {
                for (j = 0; j < cols; j++) printf("%.3f  ", matrix[i * cols + j]);
                printf("\n");
            }
            fflush(stdout);
        }
    }

    /* Broadcast dimensioni */
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Calcolo colonne locali */
    int base_cols = cols / nproc;
    int remainder = cols % nproc;
    local_cols = base_cols + (rank < remainder ? 1 : 0);

    /* Allocazioni locali */
    local_matrix = malloc(rows * local_cols * sizeof(double)); /* row-major: element (r,c) at [r*local_cols + c] */
    local_result_vector = malloc(rows * sizeof(double));
    local_prod_vector = malloc(local_cols * sizeof(double));

    /* Prepara scatterv per il vettore (contiguo) */
    int *vec_sendcounts = NULL;
    int *vec_displs = NULL;
    if (rank == 0) {
        vec_sendcounts = malloc(nproc * sizeof(int));
        vec_displs = malloc(nproc * sizeof(int));
        vec_displs[0] = 0;
        for (i = 0; i < nproc; i++) {
            int lc = base_cols + (i < remainder ? 1 : 0);
            vec_sendcounts[i] = lc;
            if (i > 0) vec_displs[i] = vec_displs[i - 1] + vec_sendcounts[i - 1];
        }
    }

    MPI_Scatterv(
        prod_vector, vec_sendcounts, vec_displs, MPI_DOUBLE,
        local_prod_vector, local_cols, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    /* --- Distribuzione colonne usando un tipo derivato su root --- */
    MPI_Datatype col_type;
    MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &col_type); /* una colonna in matrix (stride = cols) */
    MPI_Type_commit(&col_type);

    if (rank == 0) {
        int col_index = 0;
        for (int p = 0; p < nproc; p++) {
            int local_cols_p = base_cols + (p < remainder ? 1 : 0);
            if (p == 0) {
                /* Copio le colonne destinate al rank 0 nella disposizione row-major locale */
                for (int c = 0; c < local_cols_p; c++) {
                    for (int r = 0; r < rows; r++) {
                        local_matrix[r * local_cols + c] = matrix[r * cols + (col_index + c)];
                    }
                }
            } else {
                /* Invio ogni colonna con col_type: il receiver riceverà rows valori */
                for (int c = 0; c < local_cols_p; c++) {
                    MPI_Send(&matrix[col_index + c], 1, col_type, p, 0, MPI_COMM_WORLD);
                }
            }
            col_index += local_cols_p;
        }
    } else {
        /* receiver: per ogni colonna ricevo rows MPI_DOUBLE in un buffer temporaneo e la copio in row-major */
        double *colbuf = malloc(rows * sizeof(double));
        for (int c = 0; c < local_cols; c++) {
            /* ricevo i rows elementi (MPI_DOUBLE) — il root ha inviato con un derived type ma questo è compatibile */
            MPI_Recv(colbuf, rows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int r = 0; r < rows; r++) {
                local_matrix[r * local_cols + c] = colbuf[r];
            }
        }
        free(colbuf);
    }

    MPI_Type_free(&col_type);

    /* Debug printing */
    if (cols < 11 && rows < 11) {
        MPI_Barrier(MPI_COMM_WORLD);
        printf("\nRank %d local matrix:\n", rank);
        for (i = 0; i < rows; i++) {
            for (j = 0; j < local_cols; j++) printf("%.3f\t", local_matrix[i * local_cols + j]);
            printf("\n");
        }
        printf("local vector (rank %d): ", rank);
        for (j = 0; j < local_cols; j++) printf("%.2f ", local_prod_vector[j]);
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    /* calcolo parziale (local_cols passato come "cols") */
    mat_vect_prod(local_result_vector, local_matrix, rows, local_cols, local_prod_vector);

    /* reduce per sommare i contributi parziali su ogni riga */
    if (rank == 0) {
        /* result_vector già allocato nel root all'inizio */
        /* ma se non lo fosse, assicurarsi di allocarlo */
        if (result_vector == NULL) result_vector = malloc(rows * sizeof(double));
    }

    MPI_Reduce(local_result_vector, result_vector, rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    end = MPI_Wtime() - start;
    MPI_Reduce(&end, &mean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (cols < 11 && rows < 11) {
            printf("\nResult vector:\n");
            for (i = 0; i < rows; i++) printf("%f\n", result_vector[i]);
        }
        printf("\nProcs:%d, Time:%f, Matrix:%dx%d\n", nproc, mean / nproc, rows, cols);

        FILE *f = fopen("result_vector_col.txt", "a");
        if (f == NULL) {
            perror("Error file could not be opened");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f, "Procs:%d, Time:%f; Matrix:%dx%d\n", nproc, mean / nproc, rows, cols);
        fclose(f);
    }

    /* Pulizia */
    if (rank == 0) {
        free(matrix);
        free(prod_vector);
        free(result_vector);
        free(vec_sendcounts);
        free(vec_displs);
    }
    free(local_matrix);
    free(local_result_vector);
    free(local_prod_vector);

    MPI_Finalize();
    return 0;
}
