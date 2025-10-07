#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define scalable_n 1200000
#define max_iterations 5
#define scale_times 5

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    srand((unsigned int) time(0)); 
    float start,end;
    int n=scalable_n;


    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    //scale input n "scale_times" times
    for(int scale=0;scale<scale_times;scale++){

        int* sum_array = calloc(n,sizeof(int));
        int* recv_array = calloc(n/world_size,sizeof(int));
        float total_mean;
        float mean=0;
        int total_sum;
        //for each input value repeat executions "max_iterations" times to obtain mean
        for(int repeat=0;repeat<max_iterations;repeat++){
            int sum=0;
            if(rank==0){
                for(int i=0;i<n;i++){
                    sum_array[i]=rand()%3;
                } 
            }

			start=MPI_Wtime()*1000; //conversion in ms

            //sends portions of array to all procs
            MPI_Scatter(sum_array, n/world_size, MPI_INT, recv_array, n/world_size, MPI_INT, 0,MPI_COMM_WORLD);

            MPI_Barrier(MPI_COMM_WORLD);
            
            for(int i=0;i<n/world_size;i++){
                sum+=recv_array[i];
            }

            MPI_Reduce(&sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            end=MPI_Wtime()*1000-start;
        	mean+=end;
        }

        //print mean time of execution 
        MPI_Reduce(&mean, &total_mean, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank==0){
            printf("total sum is %d\n",total_sum);
            printf("Mean execution time with %d processes, %d iterations and n=%d: %.3f\n\n",world_size,max_iterations,n,total_mean/max_iterations);    
        }
        n*=2;    
        free(sum_array);
        free(recv_array);
    }

   
    MPI_Finalize();
}
