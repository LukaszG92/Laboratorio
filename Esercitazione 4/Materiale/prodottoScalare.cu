#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <time.h>

// Funzione per la gestione degli errori CUDA
void checkCuda(cudaError_t result, const char *fn, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, (unsigned int)result, cudaGetErrorString(result), fn);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(val) checkCuda((val), #val, __FILE__, __LINE__)

void prodottoCPU(float *a, float *b, float *c, int n);
__global__ void prodottoGPU(float* a, float* b, float* c, int n);

int main(void) {
	float *a_h, *b_h, *c_h, *c_d_on_h; // host data
	float *a_d, *b_d, *c_d; // device data
	int N, nBytes;
	dim3 gridDim, blockDim;
	float elapsed_gpu, elapsed_cpu;
	cudaEvent_t start_gpu, stop_gpu, start_cpu, stop_cpu;

	printf("Prodotto scalare di due vettori\n");
	printf("===============================\n");
	printf("Inserisci il numero degli elementi dei vettori: ");
	printf("\n");
	scanf("%d", &N);
	printf("Inserisci il numero di thread per blocco: ");
	scanf("%d", &blockDim.x);
	printf("\n");

	// Determinazione esatta del numero di blocchi
	gridDim = N / blockDim.x + ((N % blockDim.x) == 0 ? 0:1);

	nBytes = N * sizeof(float);
	a_h = (float *)malloc(nBytes);
	b_h = (float *)malloc(nBytes);
	c_h = (float *)malloc(nBytes);
	cudaMalloc((void **) &a_d, nBytes);
	cudaMalloc((void **) &b_d, nBytes);
	cudaMalloc((void **) &c_d, nBytes);
	c_d_on_h = (float *)malloc(nBytes);

	// Generazione casuale inizializzata mediante il tempo corrente
	srand((unsigned int) time(0));
	for (int i = 0; i < N; i++) {
		a_h[i] = rand() % 5 - 2;
		b_h[i] = rand() % 5 - 2;;
	}

	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice);

	// Azzeriamo il contenuto del vettore c
	memset(c_h, 0, nBytes);
	cudaMemset(c_d, 0, nBytes);

	// Invocazione del kernel
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu);
	printf("GridDim = %d, BlockDim = %d\n", gridDim.x, blockDim.x);
	prodottoGPU<<<gridDim, blockDim>>>(a_d, b_d, c_d, N);
	CUDA_CHECK(cudaGetLastError());
	cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost);
	float sommaGPU = 0;
	for(int i = 0; i < N; i++){
		sommaGPU += c_h[i];
	}
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&elapsed_gpu, start_gpu, stop_gpu);
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);

	cudaEventCreate(&start_cpu);
	cudaEventCreate(&stop_cpu);
	cudaEventRecord(start_cpu);
	// Calcolo somma seriale su CPU
	prodottoCPU(a_h, b_h, c_d_on_h, N);
	float sommaCPU = 0;
	for(int i = 0; i < N; i++){
		sommaCPU += c_d_on_h[i];
	}
	cudaEventRecord(stop_cpu);
	cudaEventSynchronize(stop_cpu);

	cudaEventElapsedTime(&elapsed_cpu, start_cpu, stop_cpu);
	cudaEventDestroy(start_cpu);
	cudaEventDestroy(stop_cpu);

	// Verifica che i risultati di CPU e GPU siano uguali
	// Se non stampa nulla, i due vettori sono uguali
	for (int i = 0; i <  N; i++) {
		assert( c_h[i] == c_d_on_h[i] );
	}

	if (N<20){
		for(int i = 0; i < N; i++)
			printf("a_h[%d]=%6.2f ",i, a_h[i]);
		printf("\n");
		for(int i = 0; i < N; i++)
			printf("b_h[%d]=%6.2f ",i, b_h[i]);
		printf("\n");
		for(int i = 0; i < N; i++)
			printf("c_h[%d]=%6.2f ",i, c_h[i]);
		printf("\n");
	}

	printf("Somma GPU = %f\n", sommaGPU);
	printf("Somma CPU = %f\n", sommaCPU);
	printf("time_GPU = %6.2f\n", elapsed_gpu);
	printf("time_CPU = %6.2f\n", elapsed_cpu);
	assert(sommaGPU == sommaCPU);

	free(a_h);
	free(b_h);
	free(c_h);
	free(c_d_on_h);
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	return 0;
}

// Host
void prodottoCPU(float *a, float *b, float *c, int n) {
	for(int i = 0; i < n; i++) {
		c[i] = a[i] * b[i];
	}
}

// Device
__global__ void prodottoGPU(float* a, float * b, float* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < n)
		c[index] = a[index] * b[index];
}
