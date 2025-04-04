#include <stdio.h>

__device__ float s=0;

__global__ void gInit(float* a, float* b){
 int i=threadIdx.x+blockIdx.x*blockDim.x;

 a[i]=0.01*i;
 b[i]=0.01*1.0;
}
__global__ void gScalarProduct(float* a, float* b){
 int i=threadIdx.x+blockIdx.x*blockDim.x;
/*
 __shared__ float portion[1024];

 portion[threadIdx.x]=a[i]*b[i];
 __syncthreads();

 for (int n = blockDim.x / 2; n > 0; n >>= 1){
    if (threadIdx.x < n)
      portion[threadIdx.x] += portion[threadIdx.x + n];
    __syncthreads();
  } 
  __syncthreads();
  if(threadIdx.x==0){
     atomicAdd(&s, portion[0]);
   }
   */
   atomicAdd(&s, a[i]*b[i]);
}

int main(){
 float *a, *b;

 int N=1<<20;
 cudaMalloc((void**)&a, N*sizeof(float));
 cudaMalloc((void**)&b, N*sizeof(float));

 gInit<<<N/1024,1024>>>(a,b);
 cudaDeviceSynchronize();
 
 gScalarProduct<<<N/1024,1024>>>(a,b);
 cudaDeviceSynchronize();

 float s_h;
 cudaMemcpyFromSymbol(&s_h, s, sizeof(float));
 printf("%g\n", s_h);

 return 0;
}
