#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv){
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    srand((double) time(0)); 
    float start,end;
    double sum=0;
    int n;

    for(int i=0;i<argc;i++){
        printf("Arg %d: %s\n",i,argv[i]);
    }

    if (argc > 0) {
        n = atoi(argv[1]);
    } else {
           printf("Inserire il numero di elementi da sommare: \n");
			fflush(stdout);
			scanf("%d",&n);
    }

    double *arr=malloc(n*sizeof(double));
	//generate n random numbers
	for(int i=0;i<n;i++){
		arr[i]=(rand()%5)/100.0;
	}
    start=MPI_Wtime()*1000; //conversion in ms
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    end=MPI_Wtime()*1000 - start;

    printf("Sum of n:%d numbers is %f\n",n,sum);
        

    //print time of execution 
    printf("Input n=%d, Time: %.2f\n",n, end);
    FILE *f = fopen("seq_results.txt", "a");  
    if (f == NULL) {
        perror("Error file could not be opened");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(f, "Procs:1; Time:%f; Input:%d\n",end,n);
    fclose(f);

    free(arr);

    MPI_Finalize();
}
