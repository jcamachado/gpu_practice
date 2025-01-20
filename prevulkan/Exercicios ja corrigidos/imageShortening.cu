// #include <math.h>
// #include <stdio.h>
//Sequencial
#include <iostream>
#include <FreeImage.h>
// #define N 20

// __global__ void matrixSum(int d_a[N][N], int d_b[N][N], int d_c[N][N]){
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.x * blockDim.x;

//     d_c[i][j] = d_a[i][j] + d_b[i][j];
// }
using namespace std;
int main(int argc, char **argv){
// FILE *file = fopen(argv[1], "rb");
    FreeImage_Initialise();
    //FIBITMAP *img1 = FreeImage_Load(FIF_JPEG, "imgs/teste2.jpg", JPEG_ACCURATE);
    FIBITMAP *img1 = FreeImage_Load(FIF_PNG, "imgs/teste1.png", PNG_DEFAULT);

    unsigned int wImg1 = FreeImage_GetWidth(img1); //unsigned = positive number
    unsigned int hImg1 = FreeImage_GetHeight(img1);
    cout << "Width: " << wImg1 << endl;
    cout << "Height: " << hImg1 << endl;

    RGBQUAD white = { 255, 255, 255, 0 };
    RGBQUAD pixelColor = { 0, 0, 0};

    int size = wImg1 * hImg1;
    // cout << (bool)FreeImage_HasBackgroundColor<< endl; //funciona
    //FIBITMAP* bgwhite = FreeImage_SetBackgroundColor(img1, &white); //nao funciona, retorno bool
    // BYTE curve[256];
    // for (int i = 0; i < 256; i++) {
    //     curve[i] = i;
    // }
    // for (int i = 0; i < 128; i++) {
    //     curve[i] = 255;
    // }
    // FreeImage_AdjustCurve(img1, curve, FICC_ALPHA);
    //FreeImage_SetBackgroundColor(img1, &white); //nao funciona
    // cout << FreeImage_GetTransparentIndex(img1)<< endl; //retorno -1
    //FreeImage_SetTransparentIndex(img1, 50); nao funciona
    
    //FreeImage_FillBackground(img1, &white, 0); //funciona
    //FIBITMAP* grayscale = FreeImage_ConvertToGreyscale(img1); // funciona
    //FreeImage_ConvertToGreyscale(img1); //nao funciona
    //img1 = FreeImage_RotateEx(img1, 90, 0, 0, 0, 0, FALSE); //funciona

    for (int i=0; i < wImg1; i++){
        for (int j=0; j < hImg1; j++){
            FreeImage_GetPixelColor(img1, i, j, &pixelColor);
            BYTE grayscaleColor = 0.299 * pixelColor.rgbRed + 0.587 * pixelColor.rgbGreen + 0.114 * pixelColor.rgbBlue;
            pixelColor.rgbRed = grayscaleColor;
            pixelColor.rgbGreen = grayscaleColor;
            pixelColor.rgbBlue = grayscaleColor;
            FreeImage_SetPixelColor(img1, i, j, &pixelColor);
        }
    }

    // FreeImage_SetPixelColor(img1, x, y, RGBQUAD *value);

    if (FreeImage_Save(FIF_PNG, img1, "imgs/teste1_processed.png", 0)) {
        // png successfully saved!
        cout << "Sucesso!" << endl;
    }else{
        cout << "Falha!" << endl;
        FreeImage_Unload(img1);
        FreeImage_DeInitialise();
        return 1;
    }
    FreeImage_Unload(img1);
    FreeImage_DeInitialise();


    // int n = N * N; // Arbitrei o tamanho do problema

    // // int **a = new int*[rows];
    // // int **c = new int*[rows];
    // int a[N][N], b[N][N], c[N][N];
    // for (int i = 0; i < N; i++) {
    //     // a[i] = new int[cols];
    //     for (int j = 0; j < N; j++) {
    //         a[i][j] = rand() % 100; // Generate random integer between 0 and 99
    //     }
    // }
    // // int **b = new int*[rows];
    // for (int i = 0; i < N; i++) {
    //     // b[i] = new int[cols];
    //     for (int j = 0; j < N; j++) {
    //         b[i][j] = rand() % 100; // Generate random integer between 0 and 99
    //     }
    // }

    // int (*d_a)[N], (*d_b)[N], (*d_c)[N]; 

    // cudaMalloc((void**) &d_a, sizeof(int)*n);
    // cudaMalloc((void**) &d_b, sizeof(int)*n);
    // cudaMalloc((void**) &d_c, sizeof(int)*n);

    // cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_b, b, sizeof(int) * n, cudaMemcpyHostToDevice);
    // dim3 threadsPerBlock(16, 16);
    // int blocks = 1 + ((N - 1) / threadsPerBlock.x);

    // matrixSum<<<blocks,threadsPerBlock>>>(d_a, d_b, d_c);
    // cudaMemcpy(c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%d - %d\n", i, j);

    //         printf("a[%d][%d] %d + b[%d][%d] %d= %d\n", i, j, a[i][j], i, j, b[i][j], c[i][j]);
    //    }
    // }
    // cudaFree(d_a);
    // cudaFree(d_b);
    // cudaFree(d_c);
    // return 0;
}