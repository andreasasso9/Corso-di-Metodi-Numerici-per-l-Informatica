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
    double *matrix,*result_vector,*prod_vector;
    float start,end,mean;

    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
   
    //Input data from argv or request user
    if (argc > 1){
            rows=atoi(argv[1]); //convert to int
            cols=atoi(argv[2]);
    }else{
        printf("Insert rows number = \n"); 
        fflush(stdout);
        scanf("%d",&rows); 
                
        printf("Insert columns number = \n"); 
        fflush(stdout);
        scanf("%d",&cols);
    }
        
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
     
    
    //Each process calculate the local result vector
    MPI_Barrier(MPI_COMM_WORLD);
    start=MPI_Wtime();
    mat_vect_prod(result_vector,matrix,rows,cols,prod_vector);
        
    end=MPI_Wtime()-start;

    if(rank==0){ 
        //Print result vector if small enough
        if (cols<11 && rows<11){
            for(i = 0; i < rows; i++){
                printf("%f\n", result_vector[i]);
            }
        }
        printf("\n");


        //Print execution time
        FILE *f = fopen("result_vector_seq.txt", "a");  
        if (f == NULL) {
                perror("Error file could not be opened");
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fprintf(f,"Procs:%d, Time: %f, Matrix: %dx%d\n", nproc,end,rows,cols);       
        printf("Procs:%d, Time: %f; Matrix: %dx%d\n", nproc,end,rows,cols);       
        fclose(f);
    }

    MPI_Finalize (); /* Disattiva MPI */
    return 0;  
}
