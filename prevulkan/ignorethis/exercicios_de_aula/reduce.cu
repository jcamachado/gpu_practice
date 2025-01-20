#include <math.h>
#include <stdio.h>

#define THREADS 20 // 1000 threads per block
#define ARRAYSIZE 20

__global__ void reduceSum(int *data_in, int *data_out){
    extern __shared__ int s_data[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    s_data[tid] = data_in[tid];
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
        if (tid < stride){
            s_data[tid] += s_data[tid+stride];
        }
    __syncthreads();
    }
    if(tid == 0){
        data_out[tid] = s_data[0];
    }
}

int main(){
    int *a, *d_a, lastElement, *d_out;
    int new_array_size = ARRAYSIZE;

    a = (int*) malloc(sizeof(int) * ARRAYSIZE);    

    for(int i = 0; i < ARRAYSIZE; i++){
        a[i] = i;
    }
    printf("\nEntrada\n");
    for(int i = 0; i < ARRAYSIZE; i++){
        printf("%d ", a[i]);
    }
    if (ARRAYSIZE % 2 != 0){
        new_array_size--;
        lastElement = a[new_array_size];
        a = (int*) realloc(a, sizeof(int) * new_array_size);  
    }

    printf("\nEntrada normalizada\n");
    
    for(int i = 0; i < new_array_size; i++){
        printf("%d ", a[i]);
    }       

    cudaMalloc((void**) &d_a, sizeof(int) * new_array_size);
    cudaMemcpy(d_a, a, sizeof(int) * new_array_size, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**) &d_out, sizeof(int));
    // cudaMemcpy(d_out, 0, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));
    int numberOfBlocks = ceil((float) new_array_size/THREADS);
    printf("\nblocks: %d\n", numberOfBlocks);
    reduceSum<<<numberOfBlocks,THREADS, new_array_size*sizeof(int)>>>(d_a, d_out);
    cudaMemcpy(a, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSaida: \n");
    a[0] = a[0] + lastElement;
    printf("soma = %d ", a[0]);
    
    free(a);
    cudaFree(d_a);
    cudaFree(d_out);
    printf("\nFIM");

    return 0;
}