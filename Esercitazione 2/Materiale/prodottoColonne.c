#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define ROWS 40000
#define COLS 40000

int main(int argc, char **argv) {
    int rank, nproc;
    double *matrix, *local_matrix, *vector, *result, *local_result;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    int local_COLS = COLS / nproc;
    int remainder = COLS % nproc;
    if (rank < remainder) local_COLS++;
    
    int vec_displacement = rank * (COLS / nproc) + (rank < remainder ? rank : remainder);
    
    vector = (double*)malloc(COLS * sizeof(double));
    local_matrix = (double*)malloc(ROWS * local_COLS * sizeof(double));
    local_result = (double*)malloc(ROWS * sizeof(double));
    
    if (rank == 0) {
        matrix = (double*)malloc(ROWS * COLS * sizeof(double));
        result = (double*)malloc(ROWS * sizeof(double));

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                matrix[j * ROWS + i] = i;
            }
        }
                
        for (int j = 0; j < COLS; j++) {
            vector[j] = 1;
        }
    }

    if(ROWS <= 10 && COLS <= 10) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%f ", matrix[i * ROWS + j]);
            }
        printf("\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    MPI_Bcast(vector, COLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int *sendcounts = (int*)malloc(nproc * sizeof(int));
    int *displs = (int*)malloc(nproc * sizeof(int));
    
    int disp = 0;
    for (int i = 0; i < nproc; i++) {
        int cols_for_proc = COLS / nproc;
        if (i < remainder) cols_for_proc++;
        
        sendcounts[i] = cols_for_proc * ROWS;
        displs[i] = disp * ROWS;
        disp += cols_for_proc;
    }
    
    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, 
                 local_matrix, sendcounts[rank], MPI_DOUBLE, 
                 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < ROWS; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_COLS; j++) {
            local_result[i] += local_matrix[j * ROWS + i] * vector[vec_displacement + j];
        }
    }
    
    MPI_Reduce(local_result, result, ROWS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nRisultati:\n");
        printf("Dimensioni matrice: %d x %d\n", ROWS, COLS);
        printf("Numero processori: %d\n", nproc);
        printf("Primi 5 elementi del risultato: %f %f %f %f %f\n",
                result[0], result[1], result[2], result[3], result[4]);
        printf("Tempo di esecuzione: %f secondi\n", end_time - start_time);
    }
    
    free(local_matrix);
    free(vector);
    free(local_result);
    free(sendcounts);
    free(displs);
    
    if (rank == 0) {
        free(matrix);
        free(result);
    }
    
    MPI_Finalize();
    return 0;
}