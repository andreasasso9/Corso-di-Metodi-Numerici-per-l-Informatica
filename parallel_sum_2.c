#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define scalable_n 1600000
#define max_iterations 5
#define scale_times 5

void allReduce(int* local_sum, int rank, int world_size) {
	for (int step = 1; step < world_size; step++) {
		int partner = rank ^ step; // XOR to find partner
		int send_sum = *local_sum;
		int recv_sum = 0;
		int counter = 1;

		MPI_Sendrecv(&send_sum, 1, MPI_INT, partner, 0,
			&recv_sum, 1, MPI_INT, partner, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		*local_sum += recv_sum;
		counter*=2;
		if (counter>=world_size) 
			break; // Stop if all processes have contributed
	}
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    srand((unsigned int) time(0)); 
    float start,end;
    int n=scalable_n;


    int world_size_global;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_global);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	

	int world_size=1;
	for (int i=0; i<world_size_global; i++) {
		world_size*=2;
		if (world_size > world_size_global) {
			world_size/=2;
			break;
		}
	}

	// Create a new communicator with only the required number of processes and end the others
	MPI_Comm new_comm;
	MPI_Comm_split(MPI_COMM_WORLD, rank < world_size ? 0 : MPI_UNDEFINED, rank, &new_comm);
	if (rank >= world_size) {
		MPI_Finalize();
		return 0;
	}
	
    //scale input n "scale_times" times
    for(int scale=0;scale<scale_times;scale++){

        int* sum_array = calloc(n,sizeof(int));
        int* recv_array = calloc(n/world_size,sizeof(int));
        float total_mean;
        float mean=0;
		int sum=0;
        
        //for each input value repeat executions "max_iterations" times to obtain mean
        for(int repeat=0;repeat<max_iterations;repeat++){
            
            if(rank==0){
                for(int i=0;i<n;i++){
                    sum_array[i]=rand()%3;
                } 
            }

			start=MPI_Wtime()*1000; //conversion in ms

            //sends portions of array to all procs
            MPI_Scatter(sum_array, n/world_size, MPI_INT, recv_array, n/world_size, MPI_INT, 0,new_comm);

            MPI_Barrier(new_comm);

            for(int i=0;i<n/world_size;i++){
                sum+=recv_array[i];
            }

            allReduce(&sum, rank, world_size);
            MPI_Barrier(new_comm);

        	end=MPI_Wtime()*1000-start;
            mean+=end;
        }

		mean/=max_iterations;

        //print mean time of execution 
        MPI_Reduce(&mean, &total_mean, 1, MPI_FLOAT, MPI_SUM, 0, new_comm);

		
		if (rank == 0) {
			printf("\n\nMean execution time with %d processes, %d iterations and n=%d: %.3f\n",world_size,max_iterations,n,total_mean/world_size);
			printf("total sum is %d\n", sum);
		}
        
        n*=2;    
        free(sum_array);
        free(recv_array);
    }

   
    MPI_Finalize();
	return 0;
}
