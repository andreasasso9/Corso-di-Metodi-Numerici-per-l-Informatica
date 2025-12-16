#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

__global__ void gpuVectProduct(double *vect1, double *vect2, double *result, int dim, int steps);
double serialVectProduct(double *vect1, double *vect2, int dim);
void printvector(double *vect, int dim);

int main() {
  int dim;
  double *nums1, *nums2, *prodResult;
  double *deviceNums1, *deviceNums2, *deviceProdResult;
  float serialElapsed, gpuElapsed;
  dim3 gridDim, blockDim;

  printf("Enter vectors' dimension\n");
  scanf("%d", &dim);

  nums1=(double *) calloc(dim, sizeof(double));
  nums2=(double *) calloc(dim, sizeof(double));

  int nBytes=dim*sizeof(double);

  cudaMalloc((void**) &deviceNums1, nBytes);
  cudaMalloc((void**) &deviceNums2, nBytes);
  


  srand((unsigned int) time(0));

  for (int i=0; i<dim; i++) {
    nums1[i]=rand()%10;
    nums2[i]=rand()%10;
  }

  cudaMemcpy(deviceNums1, nums1, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceNums2, nums2, nBytes, cudaMemcpyHostToDevice);

  /** Compute Capability 7.5 **/
  /*
    MaxThread = 1024
    MaxThreadBlock = 1024
    MaxBlock = 16
    BlockDim = 1024/16 = 64
    RegistersPerThread = 22
    TotalUsedRegisters = 22528
    MaxUsableRegisters = 64k 
    MaxGridDimensionality = 2^31-1
    MaxN < 2^31-1 * 64

  */
  blockDim.x = 64;
  gridDim.x=dim/blockDim.x+((dim%blockDim.x)==0?0:1);
  printf("Grid dim: %d\n", gridDim.x);

  prodResult=(double *) calloc(gridDim.x, sizeof(double));
  cudaMalloc((void**) &deviceProdResult, gridDim.x*sizeof(double));

  cudaEvent_t serialStart, serialEnd, gpuStart, gpuEnd;
  cudaEventCreate(&serialStart);
  cudaEventCreate(&serialEnd);
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuEnd);


  /*
    Serial start
  */
  cudaEventRecord(serialStart);
  double serialResult=0;

  serialResult = serialVectProduct(nums1, nums2, dim);

  cudaEventRecord(serialEnd);
  cudaEventSynchronize(serialEnd);

  cudaEventElapsedTime(&serialElapsed, serialStart, serialEnd);
  cudaEventDestroy(serialStart);
  cudaEventDestroy(serialEnd);
  printf("Serial time=%f\n", serialElapsed);
  /*
    Serial end
  */


  /*
    GPU start
  */
  cudaEventRecord(gpuStart);

  int steps=log(blockDim.x)/log(2);
  printf("Steps: %d\n", steps);
  gpuVectProduct<<<gridDim, blockDim, blockDim.x*sizeof(double)>>>(deviceNums1, deviceNums2, deviceProdResult, dim, steps);

  cudaMemcpy(prodResult, deviceProdResult, gridDim.x*sizeof(double), cudaMemcpyDeviceToHost);

  double finalResult=0;
  for (int i=0; i<gridDim.x; i++)
    finalResult += prodResult[i];

  cudaEventRecord(gpuEnd);
  cudaEventSynchronize(gpuEnd);

  
  // tempo tra i due eventi in millisecondi
  cudaEventElapsedTime(&gpuElapsed, gpuStart, gpuEnd);
  cudaEventDestroy(gpuStart);
  cudaEventDestroy(gpuEnd);
  printf("GPU time=%f\n", gpuElapsed);
  /*
    GPU end
  */

  if (dim <= 10) {
    printf("Vector 1: ");
    printvector(nums1, dim);

    printf("Vector 2: ");
    printvector(nums2, dim);
  }

  printf("Gpu Result: %f\nSerial result: %f\n", finalResult, serialResult);

  free(nums1);
  free(nums2);
  free(prodResult);

  cudaFree(deviceNums1);
  cudaFree(deviceNums2);
  cudaFree(deviceProdResult);


return 0;
}

__global__ void gpuVectProduct(double *vect1, double *vect2, double *result, int dim, int steps) {
  int id=threadIdx.x + blockIdx.x*blockDim.x;
  extern __shared__ double prods[];

  if (id < dim)
    prods[threadIdx.x] = vect1[id] * vect2[id];
  
  __syncthreads();

  //divergent sum
  int dist=blockDim.x;
  for (int i=0; i<steps; i++) {
    dist/=2;
    if (threadIdx.x < dist)
      prods[threadIdx.x] += prods[threadIdx.x+dist];


    __syncthreads();
    
  }
  __syncthreads();
  if (threadIdx.x == 0)
    result[blockIdx.x] = prods[0];

}

double serialVectProduct(double *vect1, double *vect2 ,int dim) {
  double result=0;
  for (int i=0; i<dim; i++)
    result += vect1[i]*vect2[i];

  return result;
}

void printvector(double *vect, int dim) {
  for (int i=0; i<dim; i++)
    printf("%f ", vect[i]);
  printf("\n");
}
