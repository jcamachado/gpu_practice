#include <math.h>
#include <stdio.h>
#define N 20

__global__ void matrixSum(int d_a[N][N], int d_b[N][N], int d_c[N][N]){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.x * blockDim.x;

    d_c[i][j] = d_a[i][j] + d_b[i][j];
}

int main(){
    int n = N * N; // Arbitrei o tamanho do problema

    // int **a = new int*[rows];
    // int **c = new int*[rows];
    int a[N][N], b[N][N], c[N][N];
    for (int i = 0; i < N; i++) {
        // a[i] = new int[cols];
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 100; // Generate random integer between 0 and 99
        }
    }

    // int **b = new int*[rows];
    for (int i = 0; i < N; i++) {
        // b[i] = new int[cols];
        for (int j = 0; j < N; j++) {
            b[i][j] = rand() % 100; // Generate random integer between 0 and 99
        }
    }

    int (*d_a)[N], (*d_b)[N], (*d_c)[N]; 

    cudaMalloc((void**) &d_a, sizeof(int)*n);
    cudaMalloc((void**) &d_b, sizeof(int)*n);
    cudaMalloc((void**) &d_c, sizeof(int)*n);

    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * n, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16);
    int blocks = 1 + ((N - 1) / threadsPerBlock.x);

    matrixSum<<<blocks,threadsPerBlock>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d - %d\n", i, j);

            printf("a[%d][%d] %d + b[%d][%d] %d= %d\n", i, j, a[i][j], i, j, b[i][j], c[i][j]);
       }
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}