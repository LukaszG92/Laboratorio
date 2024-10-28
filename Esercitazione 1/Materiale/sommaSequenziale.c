#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 1600000000

int main(int argc, char **argv) {
    int *numbers;
    long total_sum = 0;
    double start_time, end_time; 
    
    MPI_Init(&argc, &argv);
    
    numbers = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (long i = 0; i < ARRAY_SIZE; i++) {
        numbers[i] = 1;
    }
    
    start_time = MPI_Wtime();
    
    for (long i = 0; i < ARRAY_SIZE; i++) {
        total_sum += numbers[i];
    }
    
    end_time = MPI_Wtime() - start_time;
    
    printf("Total sum: %ld\n", total_sum);
    printf("Time taken: %f seconds\n", end_time);
    
    free(numbers);
    
    MPI_Finalize();
    return 0;
}