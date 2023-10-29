// #include <math.h>
// #include <stdio.h>
#include <iostream>
#include <FreeImage.h>
// #define N 20

__global__ void imgToGrayScale(unsigned char* d_img, int width, int height){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int index = i + j * width;

    if (i < width && j < height){
        unsigned char r = d_img[3 * index];
        unsigned char g = d_img[3 * index + 1];
        unsigned char b = d_img[3 * index + 2];
        BYTE gray = 0.299f * r + 0.587f * g + 0.114f * b;

        d_img[3 * index] = gray;
        d_img[3 * index + 1] = gray;
        d_img[3 * index + 2] = gray;
    }
}
using namespace std;
int main(int argc, char **argv){
// FILE *file = fopen(argv[1], "rb");
    FreeImage_Initialise();
    //FIBITMAP *img1 = FreeImage_Load(FIF_JPEG, "imgs/teste2.jpg", JPEG_ACCURATE);
    FIBITMAP *img1 = FreeImage_Load(FIF_PNG, "imgs/teste1.png", PNG_DEFAULT);
    if (!img1) {
        std::cerr << "Failed to load image file" << std::endl;
        return 1;
    }
    unsigned int wImg1 = FreeImage_GetWidth(img1); //unsigned = positive number
    unsigned int hImg1 = FreeImage_GetHeight(img1);
    cout << "Width: " << wImg1 << endl;
    cout << "Height: " << hImg1 << endl;


    int imgSize = wImg1 * hImg1;
    unsigned char* imgArray = FreeImage_GetBits(img1);
    unsigned char* d_image; 

    size_t imgRGBSize = imgSize * 3 * sizeof(unsigned char);
    cudaMalloc((void**)&d_image, imgRGBSize);
    cudaMemcpy(d_image, imgArray, imgRGBSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((wImg1 + threadsPerBlock.x - 1) / threadsPerBlock.x, (hImg1 + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // int blocks = 1 + ((imgSize - 1) / (16*16));

    imgToGrayScale<<<blocks,threadsPerBlock>>>(d_image, wImg1, hImg1);
    cudaMemcpy(imgArray, d_image,  imgRGBSize, cudaMemcpyDeviceToHost);
    FIBITMAP* newImage = FreeImage_ConvertFromRawBits(imgArray, wImg1, hImg1, wImg1 * 3, 16, 0xFF0000, 0x00FF00, 0x0000FF, false);

    if (FreeImage_Save(FIF_PNG, newImage, "imgs/teste1_processed.png", 0)) {
        // png successfully saved!
        cout << "Sucesso!" << endl;
    }else{
        cout << "Falha!" << endl;
        FreeImage_DeInitialise();
        return 1;
    }
    FreeImage_Unload(img1);
    FreeImage_DeInitialise();

    cudaFree(d_image);
    // cudaFree(d_b);
    // cudaFree(d_c);
    // return 0;
}