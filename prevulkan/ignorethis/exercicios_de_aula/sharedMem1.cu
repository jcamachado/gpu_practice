#include <math.h>
#include <stdio.h>

#define THREADS 8 // 1000 threads per block
#define ARRAYSIZE 8

// Professor, nao lembro do exemplo que o senhor fez em aula, e nao
// consegui achar o seu slide, entao inventei um exemplo para testar 


// Um teste de troca de elementos pelo seu corresponde na outra metade do vetor
// apos se dobrar)
// exemplo: um array de tamanho N
// a[0] *= 2
// se index < n/2
// a[index] = a[(index + n/2)+1] //em caso de impar, o elemento do meio nao sera trocado
// ex:  input:[0, 1, 5, 3, 7]
//      output:[6, 14, 10, 0, 2]
__global__ void swapElement(int* data){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shared[ARRAYSIZE];
    int aux;
    shared[threadIdx.x] = data[threadIdx.x];
    shared[index] *= 2;
    __syncthreads(); //Se remover, o comportamento sera indeterminado
    if (index < ARRAYSIZE/2)
        aux = shared[index];
        shared[index] = shared[(index + (ARRAYSIZE+1)/2)];
        shared[(index + (ARRAYSIZE+1)/2)] = aux;
    __syncthreads();
    data[threadIdx.x] = shared[threadIdx.x];
}

int main(){
    int *a, *d_a;

    a = (int*) malloc(sizeof(int) * ARRAYSIZE);    
    for(int i = 0; i < ARRAYSIZE; i++){
        a[i] = i;
    }

    printf("\nTeste Entrada\n");
    
    printf("%d ", a[0]);
    for(int i = 0; i < ARRAYSIZE; i++){
        printf("%d ", a[i]);
    }
    cudaMalloc((void**) &d_a, sizeof(int)*ARRAYSIZE);
    cudaMemcpy(d_a, a, sizeof(int) * ARRAYSIZE, cudaMemcpyHostToDevice);

    swapElement<<<1,THREADS>>>(d_a);

    cudaMemcpy(a, d_a, sizeof(int) * ARRAYSIZE, cudaMemcpyDeviceToHost);

    printf("\nTeste saida: \n");
    
    // for(int i = 0; i < ARRAYSIZE; i++){
    //     //first element is 
    //     if(i % 10 == 0){
    //         printf("\n");
    //     }
    //     printf("%d ", a[i]);
        
    // }
    printf("%d ", a[0]);
    for(int i = 0; i < ARRAYSIZE; i++){
        printf("%d ", a[i]);
        
    }

    free(a);
    cudaFree(d_a);
    return 0;
}