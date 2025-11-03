#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

int main (int argc, char **argv)
{           
    
    int i,j;
    int rank,world_size;
    double *matrix = NULL;              // Global matrix A (only on Rank 0)
    double *local_matrix = NULL;        // Temporary full row strip A_i (only on column 0 processes)
    double *local_result_vector = NULL; // Local resulting vector block (b_i block)
    double *result_vector = NULL;       // Global result vector b (only on Rank 0)
    double *prod_vector = NULL;         // Global input vector x (only on Rank 0)
    double *local_prod_vector = NULL;   // Local input vector block x_j
    double start, end, mean;           // Timing variables
    int rows,cols;                     // Global matrix dimensions
        
    /* MPI Cartesian Topology and Communicator Setup */
    MPI_Comm grid_comm,col_comm,row_comm; /* Communicators: 2D grid, vertical (column) strips, horizontal (row) strips */
    int dim=2; /* Dimension of the grid (2D) */
    int ndim[2], period[2], reorder; /* ndim: stores calculated P_rows, P_cols */
    int grid_coords[2],belongs[2];   // grid_coords: process location (row, column) in the grid
    int grid_id,row_id,col_id;       // Local ranks within the communicators
    int col_coords,row_coords;       // Coordinates (indices) within the sub-communicators

    MPI_Status info;
    /* Initialize MPI environment */
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    // Reads dimensions or prompts for them
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

        // Allocate global matrix and vectors
        matrix = malloc(rows * cols * sizeof(double));
        prod_vector = malloc(cols * sizeof(double));
        result_vector = malloc(rows * sizeof(double));

        // Initialize input vector x[j] = j
        for (j = 0; j < cols; j++)
            prod_vector[j] = j;

        // Initialize matrix A[i][j] = counter (sequential fill)
        int cont=0;
        for (i = 0; i < rows; i++) {
            for (j = 0; j < cols; j++) {
                    matrix[i * cols + j] = cont;
                    cont++;
            }
        }

        // Debug print of global data for small matrices
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

    /* Broadcast global dimensions to all processes */
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Calculates grid dimensions P_rows x P_cols
    int proc_rows = (int)floor(sqrt((double)world_size)); // Number of grid rows
    int proc_cols = world_size / proc_rows; // Number of grid columns

    ndim[0] = proc_rows; 
    ndim[1]= proc_cols;
    period[0] = 0; period[1]=0; // Non-periodic boundary conditions
    reorder = 0; // Disable rank reordering

    // Create the main 2D grid communicator
    MPI_Cart_create(MPI_COMM_WORLD, dim, ndim, period, reorder,&grid_comm);
    MPI_Comm_rank(grid_comm, &grid_id);
    // Determine the (row, col) coordinates of the process in the grid
    MPI_Cart_coords(grid_comm, grid_id, dim, grid_coords);

    /* Create vertical (column) sub-communicators */
    belongs[0] = 1; // Keep rows separate
    belongs[1] = 0; // Merge processes in the column dimension
    MPI_Cart_sub(grid_comm, belongs, &col_comm);
    MPI_Comm_rank(col_comm, &col_id);
    MPI_Cart_coords(col_comm, col_id, 1, &col_coords); // col_coords is the process's row index (0 to P_rows-1)

    /* Create horizontal (row) sub-communicators */
    belongs[0] = 0; // Merge processes in the row dimension
    belongs[1] = 1; // Keep columns separate
    MPI_Cart_sub(grid_comm, belongs, &row_comm);
    MPI_Comm_rank(row_comm, &row_id);
    MPI_Cart_coords(row_comm, row_id, 1, &row_coords); // row_coords is the process's column index (0 to P_cols-1)

    // Calculate local block dimensions
    int local_rows = rows / proc_rows; 
    int local_cols = cols / proc_cols; 
    int local_matrix_size = local_rows * cols; // Size of the full row strip

    // Allocate temporary space for the full row strip (A_i)
    local_matrix = calloc(local_matrix_size, sizeof(double));
    // Allocate space for the local result vector part (b_i partial)
    local_result_vector = malloc(local_rows * sizeof(double));
    local_prod_vector = malloc(local_cols * sizeof(double));
    
    // This distributes the matrix vertically to the first column of processes (where row_coords == 0)
    if (row_coords == 0) { // Condition ensures only processes in the FIRST COLUMN (row_coords == 0) receive data
        MPI_Scatter(
        matrix, local_matrix_size, MPI_DOUBLE,  // Send count is the size of the full row strip
        local_matrix, local_matrix_size, MPI_DOUBLE, // Receive into the temporary row strip buffer
        0, col_comm // Use the vertical communicator (col_comm)
        );    
    }
    if (col_coords == 0) { // Processes in the FIRST ROW (col_coords == 0) get the input vector block
        MPI_Scatter(
        prod_vector, local_cols, MPI_DOUBLE,
        local_prod_vector, local_cols, MPI_DOUBLE,
        0, row_comm); // Use the horizontal communicator (row_comm)
    }

    // Broadcast the input vector block to all processes in the column
    MPI_Bcast(
        local_prod_vector, local_cols, MPI_DOUBLE,
        0, col_comm);

    // Allocate space for the final local block A_i,j
    double *local_block = calloc(local_rows * local_cols, sizeof(double));
    
    // Define a derived datatype for a single column of the block
    MPI_Datatype col_type;
    // MPI_Type_vector(count, blocklength, stride, oldtype, newtype)
    // This type selects local_rows elements, spaced by 'cols' (the full row length)
    MPI_Type_vector(local_rows, 1, cols, MPI_DOUBLE, &col_type); 
    MPI_Type_commit(&col_type);



    // Only processes that received the full strip (row_coords == 0) act as senders in their row_comm
    if (row_coords == 0) {
        int col_index = 0; // Tracks the starting column index within the local_matrix strip
        for (int p = 0; p < proc_cols; p++) { // Iterate over all process columns
            if (p == 0) {
                /* Copy the first block (for itself) from the strip to the final block buffer */
                for (int c = 0; c < local_cols; c++) {
                    for (int r = 0; r < local_rows; r++) {
                        // Copying column by column into row-major local_block
                        local_block[r * local_cols + c] = local_matrix[r * cols + (col_index + c)];
                    }
                }
            } else {
                /* Send each block column using the derived type */
                for (int c = 0; c < local_cols; c++) {
                    // Send 1 item of type 'col_type' (which is actually local_rows doubles)
                    MPI_Send(&local_matrix[col_index + c], 1, col_type, p, 0, row_comm);
                }
            }
            col_index += local_cols; // Move to the start of the next column block
        }
    } else {
        /* Receiver processes (row_coords > 0) */
        // Allocate buffer to receive one column block
        double *colbuf = malloc(local_rows * sizeof(double));
        for (int c = 0; c < local_cols; c++) {
            /* Receive local_rows elements (one column of the block) */
            MPI_Recv(colbuf, local_rows, MPI_DOUBLE, 0, 0, row_comm, MPI_STATUS_IGNORE);
            for (int r = 0; r < local_rows; r++) {
                // Copy from temporary column buffer to the final row-major block
                local_block[r * local_cols + c] = colbuf[r];
            }
        }
        free(colbuf);
    }



        MPI_Type_free(&col_type); // Free the custom datatype
        fflush(stdout);
        // Debug print of the final local block A_i,j
        printf("Rank %d:%d local block:\n", grid_coords[0],grid_coords[1]);
        for (i = 0; i < local_rows; i++) {  
            for (j = 0; j < local_cols; j++)
                printf("%.3f  ", local_block[i * local_cols + j]);
            printf("\n\n");
        }
        printf("\nlocal vector (rank %d:%d): ", grid_coords[0],grid_coords[1]);
        for (j = 0; j < local_cols; j++)
            printf("%.2f ", local_prod_vector[j]);
        printf("\n\n");
        fflush(stdout);
    
    if (rank==0)
    {
        free(matrix);
        free(prod_vector);
        free(result_vector);
    }
    free(local_matrix);
    free(local_block);
    
    MPI_Finalize(); // Terminate the MPI environment

    return 0;
}