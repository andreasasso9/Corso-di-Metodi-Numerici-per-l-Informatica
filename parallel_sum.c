#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define scalable_n 1200000
#define max_iterations 5
#define scale_times 5


void reduce(double *sum, int rank, int world) {
    int step = 2;
    int distance = 1; 
    while (distance < world) {
        if ((rank % step) != 0) {
                MPI_Send(sum, 1, MPI_DOUBLE, rank-distance, 0, MPI_COMM_WORLD);
                break; // chi manda esce
        } else {
            double received = 0.0;
            MPI_Recv(&received, 1, MPI_DOUBLE, rank+distance,0 , MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            *sum += received;
           }
            distance=step;
            step*=2;
        }
    
        
}



int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    srand((double) time(0)); 
    float start,end;
    int n=scalable_n;


    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* final_sum_array = calloc(max_iterations,sizeof(double));//output array

    //scale input n "scale_times" times
    for(int scale=0;scale<scale_times;scale++){

        double* sum_array = calloc(n,sizeof(double));
        double* recv_array = calloc(n/world_size,sizeof(double));
        float total_mean;
        float mean=0;

        //for each input value repeat executions "max_iterations" times to obtain mean
        for(int repeat=0;repeat<max_iterations;repeat++){
            if(rank==0){
                for(int i=0;i<n;i++){
                    sum_array[i]=(rand()%5)/100.0;
                } 
            }
            double sum=0;
            MPI_Barrier(MPI_COMM_WORLD);
			start=MPI_Wtime()*1000; //conversion in ms

            //sends portions of array to all procs
            MPI_Scatter(sum_array, n/world_size, MPI_DOUBLE, recv_array, n/world_size, MPI_DOUBLE, 0,MPI_COMM_WORLD);

            
            for(int i=0;i<n/world_size;i++){
                sum+=recv_array[i];
            }
            reduce(&sum,rank,world_size); //all procs reduce sum to proc 0
            MPI_Barrier(MPI_COMM_WORLD);
            end=MPI_Wtime()*1000-start;
        	mean+=end;
            final_sum_array[repeat]=sum;
        }
        

       
        //print mean time of execution 
        MPI_Reduce(&mean, &total_mean, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank==0){
             //print sums
            for(int i=0;i<max_iterations;i++){
                printf("sum at iteration %d is %.2f\n",i,final_sum_array[i]);
            }

            printf("Mean execution time with %d processes, %d iterations and n=%d: %.2f\n\n",world_size,max_iterations,n,total_mean/max_iterations);    
        }
        n*=2;    
        free(sum_array);
        free(recv_array);
    }

   
    MPI_Finalize();
}

