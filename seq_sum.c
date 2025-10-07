#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define scalable_n 1600000
#define max_iterations 5
#define scale_times 5

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    srand((double) time(0)); 
    float start,end;
    int n=scalable_n;


    //scale input n "scale_times" times
    for(int scale=0;scale<scale_times;scale++){
        float mean=0;
        //for each input value repeat executions "max_iterations" times to obtain mean
        for(int repeat=0;repeat<max_iterations;repeat++){
            double sum=0;
			double *arr=malloc(n*sizeof(double));
			//generate n random numbers
			for(int i=0;i<n;i++){
				arr[i]=(rand()%3)/10.0;
			}

            start=MPI_Wtime()*1000; //conversion in ms
            for(int i=0;i<n;i++){
                sum+=arr[i];
            }
            end=MPI_Wtime()*1000;

            printf("Sum %d with %d numbers is %f\n",repeat,n,sum);
            mean+=(end-start);
        }

        //print mean time of execution 
        printf("Mean execution time with %d iterations and n=%d: %.3f\n\n",max_iterations,n,mean/max_iterations);    
        n*=2;    
    }

    MPI_Finalize();
}
