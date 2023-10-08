#include <math.h>
#include <stdio.h>
#define N 20

__global__ void VecAdd(int* A, int* B, int* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    int a[N], b[N], c[N];
    int *d_a, *d_b, *d_c;

    for (int i = 0; i < N; i++) {
        // a[i] = new int[cols];
            a[i] = rand() % 100; // Generate random integer between 0 and 99
    }

    // int **b = new int*[rows];
    for (int i = 0; i < N; i++) {
        // b[i] = new int[cols];
        b[i] = rand() % 100; // Generate random integer between 0 and 99
    }

    cudaMalloc((void**) &d_a, sizeof(int)*N);
    cudaMalloc((void**) &d_b, sizeof(int)*N);
    cudaMalloc((void**) &d_c, sizeof(int)*N);

    cudaMemcpy(d_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

    VecAdd<<<1, N>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
    // Kernel invocation with N threads
    for (int i = 0; i < N; i++) {
        printf("a[%d] %d + b[%d] %d= %d\n", i, a[i], i, b[i], c[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}