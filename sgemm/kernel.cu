#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

#define BLOCK_DIM 16

/*
    Compute Matrix Multiplication
    C = alpha * A * B + beta * C

    Params:
    A -> matrix A
    B -> matrix B
    C -> matrix C
    N -> height of matrix A and matrix C
    M -> width of Matrix b and matrix C
    K -> width of matrix A and height of matrix C
    alpha -> scalar value for matrix multiplication
    beta -> scalar value for matrix summation with C
*/

__global__ void sgemm_gpu_kernel(const float* A, const float* B, float* C, int N, int M, int K, float alpha, float beta)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int i = 0; i < K; ++i)
    {
        sum += A[row * K + i] * B[i * K + col];
    }

    C[row * M + col] = alpha * sum + beta * C[row * M + col];
}

__global__ void sgemm_gpu_kernel_v2(const float* A, const float* B, float* C, int N, int M, int K, float alpha, float beta)
{
    int bid_x = blockIdx.x * blockDim.x;
    int bid_y = blockIdx.y * blockDim.y;
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float element_c = 0.f;
    __shared__ float s_tile_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float s_tile_B[BLOCK_DIM][BLOCK_DIM];

    // forward tile with tile size in matrix A
    for (int k = 0; k < K; k += BLOCK_DIM)
    {
        s_tile_A[tid_y][tid_x] = A[(bid_y + tid_y) * K + tid_x + k]; // get sub-matrix from A
        s_tile_B[tid_y][tid_x] = B[(k * BLOCK_DIM + tid_y) * N + bid_x + tid_x]; // get sub-matrix from B

        __syncthreads();

        // compute gemm operation with tiles
        for (int e = 0; e < BLOCK_DIM; e++)
        {
            element_c += s_tile_A[tid_y][e] * s_tile_B[e][tid_x];
        }

        __syncthreads();
    }

    C[(bid_y + tid_y) * N + (bid_x + tid_x)] = alpha * element_c + beta * C[(bid_y + tid_y) * N + (bid_x + tid_x)];
}

void sgemm_gpu(const float* A, const float* B, float* C, int N, int M, int K, float alpha, float beta)
{
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
    sgemm_gpu_kernel_v2<<<dimGrid,dimBlock>>>(A, B, C, N, M, K, alpha, beta);
}

void random_init(float* data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = (rand() & 0xFF) / (float)RAND_MAX;
    }
}

int main()
{
    float* A, * B, * C;
    float* d_A, * d_B, * d_C;
    int N, M, K;
    float alpha = 2.f;
    float beta = 1.f;
    N = M = K = 2048;

    // Allocate host linear memory space
    A = (float*)malloc(N * K * sizeof(float));
    B = (float*)malloc(K * M * sizeof(float));
    C = (float*)malloc(N * M * sizeof(float));

    // Allocate gpu linear memory space
    cudaMalloc((void**)&d_A, N * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * M * sizeof(float));
    cudaMalloc((void**)&d_C, N * M * sizeof(float));

    // Initialize random values in host memory space
    random_init(A, N * K);
    random_init(B, K * M);
    random_init(C, N * M);

    // Copy initial values to gpu memory
    cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Perform sgemm operation
    sgemm_gpu(d_A, d_B, d_C, N, M, K, alpha, beta);
    cudaDeviceSynchronize();

    // Terminate allocated gpu memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Terminate allocated host memory
    free(A);
    free(B);
    free(C);

    return 0;
}