#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float* h_a = 0;     // Host array a
    float* d_a;         // Device array a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float result = 0;   // Risultato finale

    srand(time(0));

    int M;
    printf("Enter vectors' dimension\n");
    scanf("%d", &M);
	
	/*
	[3, 10, 20] * [5, 10, 15] = 415
	*/

    h_a = (float *)malloc (M * sizeof (*h_a));      // Alloco h_a e lo inizializzo
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    

    for (int i=0; i<M; i++)
      h_a[i]=rand()%10;
    
    h_b = (float *)malloc (M * sizeof (*h_b));  // Alloco h_b e lo inizializzo
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    
    for (int i=0; i<M; i++)
      h_b[i]=rand()%10;
    
    cudaStat = cudaMalloc ((void**)&d_a, M*sizeof(*h_a));       // Alloco d_a
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc ((void**)&d_b, M*sizeof(*h_b));       // Alloco d_b
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);               // Creo l'handle per cublas
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_a,1,d_a,1);    // Setto h_a su d_a
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_b,1,d_b,1);    // Setto h_b su d_b
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed\n");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    float Elapsed;
    cudaEvent_t gpuStart, gpuEnd;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuEnd);

    cudaEventRecord(gpuStart);
    stat = cublasSdot(handle,M,d_a,1,d_b,1,&result);        // Calcolo il prodotto
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot\n");
        cudaFree (d_a);
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaEventRecord(gpuEnd);
    cudaEventSynchronize(gpuEnd);

    cudaEventElapsedTime(&Elapsed, gpuStart, gpuEnd);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuEnd);
    printf("Execution time=%f\n", Elapsed);
    
    printf("Risultato del prodotto --> %f\n",result);
    
    cudaFree (d_a);     // Dealloco d_a
    cudaFree (d_b);     // Dealloco d_b
    
    cublasDestroy(handle);  // Distruggo l'handle
    
    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b    
    return EXIT_SUCCESS;
}