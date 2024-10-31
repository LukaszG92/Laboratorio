#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define ARRAY_SIZE 200000000

int main(int argc, char **argv) {
    int *numbers;
    int total_sum = 0;
    double start_time, end_time; 
    
    MPI_Init(&argc, &argv);
    
    numbers = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        numbers[i] = 1;
    }
    
    start_time = MPI_Wtime();
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        total_sum += numbers[i];
    }
    
    end_time = MPI_Wtime() - start_time;
    
    printf("Total sum: %d\n", total_sum);
    printf("Time taken: %f seconds\n", end_time);
    
    free(numbers);
    
    MPI_Finalize();
    return 0;
}