#include <iostream>
#include <ctime>
#include <fstream>
#include <cstdlib>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

const int arr_size = 10;
//#define IDX2C(i, j, arr_size) (((j)*(arr_size))+(i))

int main(int argc, char** argv)
{
	float *arr1, *negative_arr1, *arr2, *arr3, *S;
	float *d_arr1, *d_negative_arr1, *d_arr2, *d_arr3, *d_S;
	
	// Выделение памяти на CPU
	arr1 = (float*)malloc(arr_size * arr_size * sizeof(float));
	negative_arr1 = (float*)malloc(arr_size * arr_size * sizeof(float));
	arr2 = (float*)malloc(arr_size * arr_size * sizeof(float));
	arr3 = (float*)malloc(arr_size * arr_size * sizeof(float));
	S = (float*)malloc(arr_size * arr_size * sizeof(float));
	
	cudaError_t cudaStat;
    cublasStatus_t cublasStatus;
    
    // Очистка GPU
    cudaDeviceReset();
	
    // Выделение памяти на GPU
    cudaStat = cudaMalloc((void**)&d_arr1, arr_size * arr_size * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf ("d_arr1 memory allocation failed\n");
		return 1;
	}
	
	cudaStat = cudaMalloc((void**)&d_negative_arr1, arr_size * arr_size * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf ("d_negative_arr1 memory allocation failed\n");
		return 1;
	}
	
    cudaStat = cudaMalloc((void**)&d_arr2, arr_size * arr_size * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf ("d_arr2 memory allocation failed\n");
		return 1;
	}
	
    cudaStat = cudaMalloc((void**)&d_arr3, arr_size * arr_size * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf ("d_arr3 memory allocation failed\n");
		return 1;
	}
	
	cudaStat = cudaMalloc((void**)&d_S, arr_size * arr_size * sizeof(float));
	if (cudaStat != cudaSuccess) {
		printf ("d_S memory allocation failed\n");
		return 1;
	}
	
	// Считаем матрицу из файла
	ifstream in("numbers1.txt"), in2("numbers1.txt");
	for (long long i = 0; i < arr_size; i++)
	{
		for (long long j = 0; j < arr_size; j++)
		{	
			in >> arr1[i*arr_size + j];
			in2 >> negative_arr1[i*arr_size + j];
			negative_arr1[i*arr_size + j] *= -1;
		}
	}
	in.close();
	in2.close();

	in.open("numbers2.txt");
	for (long long i = 0; i < arr_size; i++)
		for (long long j = 0; j < arr_size; j++)
			in >> arr2[i*arr_size + j];
	in.close();

	double start_time = clock();
	
	// Копирование данных из оперативной памяти в видеопамять
	cudaMemcpy(d_arr1, arr1, arr_size * arr_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_negative_arr1, negative_arr1, arr_size * arr_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, arr_size * arr_size * sizeof(float), cudaMemcpyHostToDevice);
	
	cublasHandle_t h;
    cublasCreate(&h);
	
	const float alpha = 1.0f, beta = 0.0f;
	
	// Евклидова норма матрицы
	// S[i*arr_size + j] = arr1[i*arr_size + j] * arr1[i*arr_size + j]
	float sqrtS = 0;
	cublasStatus = cublasSnrm2(h, arr_size * arr_size,
							   d_arr1, 1, &sqrtS);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
	{
		printf ("ERROR A x B = %d i=%d \n", cublasStatus, 0);
	    return 1;
	}
	cout << "S = " << sqrtS << endl;
	
	// sqrtS * arr1[i*arr_size + j] + arr2[i*arr_size + j]
	cublasStatus = cublasSaxpy(h, arr_size * arr_size, 
							   &sqrtS, 
							   d_arr1, 1, 
							   d_arr2, 1);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
	{
		printf ("ERROR A x B = %d i=%d \n", cublasStatus, 0);
	    return 1;
	}
	
	// arr2[i*arr_size + j] - arr1[i*arr_size + j]
	cublasStatus = cublasSaxpy(h, arr_size * arr_size, 
							   &alpha, 
							   d_arr2, 1, 
							   d_negative_arr1, 1);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) 
	{
		printf ("ERROR A x B = %d i=%d \n", cublasStatus, 0);
	    return 1;
	}
	
	// Копирование данных из видеопамяти в оперативную память
    cudaMemcpy(arr3, d_negative_arr1, arr_size * arr_size * sizeof(float), cudaMemcpyDeviceToHost);

	double end_time = clock();
	double search_time = end_time - start_time;
	cout << search_time / CLOCKS_PER_SEC << endl;

	ofstream out("out.txt");
	for (int i = 0; i < arr_size; i++)
	{
		for (int j = 0; j < arr_size; j++)
		{
			out << arr3[i*arr_size + j] << " ";
		}
		out << endl;
	}
		
	cublasDestroy(h);
	
	// освобождение памяти на CPU
    free(arr1);
	free(negative_arr1);
    free(arr2);
    free(arr3);
	free(S);
	
    // освобождение памяти на GPU
    cudaFree(d_arr1);
	cudaFree(d_negative_arr1);
    cudaFree(d_arr2);
    cudaFree(d_arr3);
	cudaFree(d_S);
        
    return 0;
}