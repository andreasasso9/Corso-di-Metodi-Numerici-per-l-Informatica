#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

__global__ void gpuVectProduct(double *vect1, double *vect2, double *result, int dim);
double serialVectProduct(double *vect1, double *vect2, int dim);
void printvector(double *vect, int dim);

int main() {
  int dim;
  double *nums1, *nums2, *prodResult;
  double *deviceNums1, *deviceNums2, *deviceProdResult;

  printf("Enter vectors' dimension\n");
  scanf("%d", &dim);

  nums1=(double *) calloc(dim, sizeof(double));
  nums2=(double *) calloc(dim, sizeof(double));
  prodResult=(double *) calloc(dim, sizeof(double));


  dim3 gridDim, blockDim;
  int nBytes=dim*sizeof(double);

  cudaMalloc((void**) &deviceNums1, nBytes);
  cudaMalloc((void**) &deviceNums2, nBytes);
  cudaMalloc((void**) &deviceProdResult, nBytes);


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
    RegistersPerThread = 12
    TotalUsedRegisters = 12288
    MaxUsableRegisters = 64k 
    MaxGridDimensionality = 2^31-1
    MaxN < 2^31-1 * 64

  */
  blockDim.x = 64;
  gridDim.x=dim/blockDim.x+((dim%blockDim.x)==0?0:1);

  cudaEvent_t serialStart, serialEnd, gpuStart, gpuEnd;
  cudaEventCreate(&serialStart);
  cudaEventCreate(&serialEnd);
  cudaEventCreate(&gpuStart);
  cudaEventCreate(&gpuEnd);

  float serialElapsed, gpuElapsed;

  /**/
  cudaEventRecord(serialStart);
  double serialResult=0;

  serialResult=serialVectProduct(nums1, nums2, dim);

  cudaEventRecord(serialEnd);
  cudaEventSynchronize(serialEnd);

  cudaEventElapsedTime(&serialElapsed, serialStart, serialEnd);
  cudaEventDestroy(serialStart);
  cudaEventDestroy(serialEnd);
  printf("Serial time=%f\n", serialElapsed);


  cudaEventRecord(gpuStart);

  gpuVectProduct<<<gridDim, blockDim>>>(deviceNums1, deviceNums2, deviceProdResult, dim);

  cudaMemcpy(prodResult, deviceProdResult, nBytes, cudaMemcpyDeviceToHost);

  double finalResult=0;
  for (int i=0; i<dim; i++)
    finalResult += prodResult[i];

  cudaEventRecord(gpuEnd);
  cudaEventSynchronize(gpuEnd);

  // tempo tra i due eventi in millisecondi
  cudaEventElapsedTime(&gpuElapsed, gpuStart, gpuEnd);
  cudaEventDestroy(gpuStart);
  cudaEventDestroy(gpuEnd);
  printf("GPU time=%f\n", gpuElapsed);

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

__global__ void gpuVectProduct(double *vect1, double *vect2, double *result,int dim) {
  int id=threadIdx.x + blockIdx.x*blockDim.x;

  if (id < dim) {
    result[id] = vect1[id] * vect2[id];
  }
}

double serialVectProduct(double *vect1, double *vect2, int dim) {
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
