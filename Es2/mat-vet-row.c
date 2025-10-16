#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h> 


/**
* Function that multiplies a matrix by a vector
*/
void mat_vect_prod(double result_vector[], double *matrix, int rows, int cols, double prod_vector[]){

    for(int i=0;i<rows;i++){
        result_vector[i]=0;
        for(int j=0;j<cols;j++){ 
            result_vector[i] += matrix[i*cols+j]* prod_vector[j];
        } 
    }    
}


int main(int argc, char **argv) {

    int i,j;
    int nproc,rank;            
    int rows,cols;                  //Matrix size
    int local_rows;            //Local matrix rows
    double *matrix,*local_matrix,*local_result_vector,*result_vector,*prod_vector;
    float start,end,mean;

    MPI_Init(&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //Root take input data then allocate and populate local matrix and product vector
    if(rank == 0){

        //Input data from argv or request user
        if (argc > 3){
            rows=atoi(argv[2]); //convert to int
            cols=atoi(argv[3]);
        }else{
            printf("Insert rows number = \n"); 
            fflush(stdout);
            scanf("%d",&rows); 
                
            printf("Insert columns number = \n"); 
            fflush(stdout);
            scanf("%d",&cols);
        }
       

        ////Local matrix rows for each proc
        local_rows = rows/nproc;  
        
        //Allocate matrix, product vector and final result vector
        matrix = malloc(rows * cols * sizeof(double));
        prod_vector = malloc(sizeof(double)*cols);
        result_vector =  malloc(sizeof(double)*rows); 
        
        //Populate product vector and matrix
        for (j=0;j<cols;j++)
                prod_vector[j]=j; 


        //Array for the matrix is populated using as the column number as displacement for each row
        for (i=0;i<rows;i++){
            for(j=0;j<cols;j++){
                if (j==0)
                    matrix[i*cols+j]= 1.0/(i+1)-1;
                else
                    matrix[i*cols+j]= 1.0/(i+1)-pow(1.0/2.0,j); 
            }
        }
        
        //Print the product vector and matrix if small
        if (cols<11 && rows<11){  
        printf("\nv: \n");   
            for (j=0;j<cols;j++)
                printf("%.2f ", prod_vector[j]);
            printf("\n\n");
        
            printf("matrix: \n"); 
            for (i=0;i<rows;i++){
                for(j=0;j<cols;j++)
                        printf("%.3f  ", matrix[i*cols+j] );
                printf("\n\n");
            }
            fflush(stdout);
        } 

    } 


    //Send rows,columns and local rows to each process so they can allocate their local matrix and result vector
    MPI_Bcast(&cols,1,MPI_INT,0,MPI_COMM_WORLD);            
    MPI_Bcast(&local_rows,1,MPI_INT,0,MPI_COMM_WORLD);      

    local_matrix = malloc(local_rows * cols * sizeof(double));
    local_result_vector = malloc(local_rows * sizeof(double));

    //if not root allocate product vector
    if(rank)
        prod_vector = malloc(sizeof(double)*cols);

    //each process populate prod_vector!
    MPI_Bcast(&prod_vector[0],cols,MPI_DOUBLE,0,MPI_COMM_WORLD);            


    //Root send set of local rows of global matrix to each process so they can populate their local one
    int local_matrix_size = local_rows*cols;
    MPI_Scatter(
        &matrix[0], local_matrix_size, MPI_DOUBLE,
        &local_matrix[0], local_matrix_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    //Print local matrix if small enough
	if (cols<11 && local_rows<11){
        printf("local matrix %d : \n", rank); 
        for(i = 0; i < local_rows; i++){
            for(j = 0; j < cols; j++)         
                printf("%.3f\t", local_matrix[i*cols+j]);
            printf("\n\n");
        }
    }
     
    
    //Each process calculate the local result vector
    MPI_Barrier(MPI_COMM_WORLD);
    start=MPI_Wtime();
    mat_vect_prod(local_result_vector,local_matrix,local_rows,cols,prod_vector);
        
    //Results are gathered by root process and printed
    MPI_Gather(&local_result_vector[0],local_rows,MPI_DOUBLE,&result_vector[0],local_rows,MPI_DOUBLE,0,MPI_COMM_WORLD);
    end=MPI_Wtime()-start;
    MPI_Reduce(&end, &mean, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank==0){ 
        //Print result vector if small enough
        if (cols<11 && local_rows<11){
            for(i = 0; i < rows; i++){
                printf("%f\n", result_vector[i]);
            }
        }
        printf("\n");


        //Print execution time
        FILE *f = fopen("result_vector_row.txt", "a");  
        if (f == NULL) {
                perror("Error file could not be opened");
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f,"Procs:%d, Time: %f, Matrix: %dx%d\n", nproc,mean/nproc,rows,cols);     
        printf("Procs:%d, Time: %f, Matrix: %dx%d\n", nproc,mean/nproc,rows,cols);       
        fclose(f);
    }

    MPI_Finalize (); /* Disattiva MPI */
    return 0;  
}
