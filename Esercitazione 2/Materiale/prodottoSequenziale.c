#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define ROWS 20000
#define COLS 20000

int main(int argc, char **argv) {
    double *matrix, *vector, *result;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    
    matrix = (double*)malloc(ROWS * COLS * sizeof(double));
    vector = (double*)malloc(COLS * sizeof(double));
    result = (double*)malloc(ROWS * sizeof(double));
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            matrix[i * COLS + j] = i;  
        }
    }
    
    for (int j = 0; j < COLS; j++) {
        vector[j] = 1.0;
    }
    
    start_time = MPI_Wtime();
    
    for (int i = 0; i < ROWS; i++) {
        result[i] = 0.0;
        for (int j = 0; j < COLS; j++) {
            result[i] += matrix[i * COLS + j] * vector[j];
        }
    }
    
    end_time = MPI_Wtime() - start_time;
    
    printf("Tempo di esecuzione: %f secondi\n", end_time);
    
    free(matrix);
    free(vector);
    free(result);
    
    MPI_Finalize();
    return 0;
}