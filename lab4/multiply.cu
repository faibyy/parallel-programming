#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

#define gpuerrchk(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        cout << "CUDA Error: " << cudaGetErrorString(error) << endl; \
        exit(1); \
    } \
}

__global__ void matrixMultiplyKernel(
    long long* A,
    long long* B,
    long long* C,
    int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        long long sum = 0;
        for (int k = 0; k < n; ++k)
            sum += A[i * n + k] * B[k * n + j];

        C[i * n + j] = sum;
    }
}

void readMatrix(const string& filename, vector<long long>& matrix, int n)
{
    ifstream file(filename);
    for (int i = 0; i < n * n; ++i)
        file >> matrix[i];
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        cout << "Usage: kernel.exe <A> <B> <block_size>\n";
        return 1;
    }
    std::vector<int> sizes = { 200, 400, 800, 1200, 1600, 2000 };
    for (int n : sizes) {
        string fileA = argv[1] + '_' + to_string(n) + ".txt";
        string fileB = argv[2] + '_' + to_string(n) + ".txt";
        int blockSize = stoi(argv[3]);

        size_t bytes = n * n * sizeof(long long);

        vector<long long> h_A(n * n);
        vector<long long> h_B(n * n);
        vector<long long> h_C(n * n);

        readMatrix(fileA, h_A, n);
        readMatrix(fileB, h_B, n);

        long long* d_A, * d_B, * d_C;

        gpuerrchk(cudaMalloc(&d_A, bytes));
        gpuerrchk(cudaMalloc(&d_B, bytes));
        gpuerrchk(cudaMalloc(&d_C, bytes));

        gpuerrchk(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
        gpuerrchk(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(blockSize, blockSize);
        dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matrixMultiplyKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, n);
        gpuerrchk(cudaPeekAtLastError());
        gpuerrchk(cudaDeviceSynchronize());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        gpuerrchk(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

        cout << "\nMatrix size: " << n << "x" << n << endl;
        cout << "Block size: " << blockSize << "x" << blockSize << endl;
        cout << "Grid size: " << blocksPerGrid.x << "x" << blocksPerGrid.y << endl;
        cout << "Execution time: " << milliseconds / 1000.0 << " seconds\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

    }
    return 0;
}