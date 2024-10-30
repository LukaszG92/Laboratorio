#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define ROWS 40000
#define COLS 40000

void prod_mat_vett(double result_vector[], double *a, int rows, int cols, double vector[]) {
    int i, j;
    for(i = 0; i < rows; i++) {
        result_vector[i] = 0;
        for(j = 0; j < cols; j++) {
            result_vector[i] += a[i * cols + j] * vector[j];
        }
    }
}

int main(int argc, char **argv) {
    int rank, nproc;
    double *matrix = NULL, *vector = NULL, *result = NULL;
    double *local_matrix = NULL, *local_result = NULL, *local_vector = NULL;
    double start_time, end_time, max_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate local columns for each process
    int local_COLS = COLS / nproc;
    int remainder = COLS % nproc;
    
    // Calculate actual columns for this process
    int my_cols = (rank < remainder) ? local_COLS + 1 : local_COLS;
    
    // Arrays for scattering
    int *sendcounts = (int*)malloc(nproc * sizeof(int));
    int *displs = (int*)malloc(nproc * sizeof(int));
    int *vector_sendcounts = (int*)malloc(nproc * sizeof(int));
    int *vector_displs = (int*)malloc(nproc * sizeof(int));

    // Calculate send counts and displacements for both matrix and vector
    int current_disp = 0;
    for(int i = 0; i < nproc; i++) {
        int cols_for_proc = (i < remainder) ? local_COLS + 1 : local_COLS;
        sendcounts[i] = cols_for_proc * ROWS;  // Total elements for this process
        displs[i] = current_disp * ROWS;       // Displacement in number of elements
        
        vector_sendcounts[i] = cols_for_proc;  // Elements of vector for this process
        vector_displs[i] = current_disp;       // Displacement in vector
        
        current_disp += cols_for_proc;
    }

    // Allocate local buffers
    local_matrix = (double*)malloc(my_cols * ROWS * sizeof(double));
    local_vector = (double*)malloc(my_cols * sizeof(double));
    local_result = (double*)malloc(ROWS * sizeof(double));

    if (rank == 0) {
        // Allocate and initialize global arrays
        matrix = (double*)malloc(ROWS * COLS * sizeof(double));
        vector = (double*)malloc(COLS * sizeof(double));
        result = (double*)malloc(ROWS * sizeof(double));

        // Initialize matrix and vector
        for (int i = 0; i < ROWS; i++) {
            for(int j = 0; j < COLS; j++) {
                matrix[i * COLS + j] = 1.0;
            }
        }
        for (int j = 0; j < COLS; j++) {
            vector[j] = 1.0;
        }
    }

    // Scatter the matrix and vector
    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE,
                 local_matrix, my_cols * ROWS, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(vector, vector_sendcounts, vector_displs, MPI_DOUBLE,
                 local_vector, my_cols, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Compute local matrix-vector product
    prod_mat_vett(local_result, local_matrix, ROWS, my_cols, local_vector);

    // Reduce results
    MPI_Reduce(local_result, result, ROWS, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime() - start_time;
    MPI_Reduce(&end_time, &max_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        fprintf(stdout, "\nExecution Summary\n");
        fprintf(stdout, "===========================\n");
        fprintf(stdout, "Number of processors: %d\n", nproc);
        fprintf(stdout, "Local computation time: %lf\n", end_time);
        fprintf(stdout, "First 5 elements of result: %f %f %f %f %f\n",
                result[0], result[1], result[2], result[3], result[4]);
        fprintf(stdout, "MPI_Reduce max time: %f\n", max_end);
    }

    // Cleanup
    free(sendcounts);
    free(displs);
    free(vector_sendcounts);
    free(vector_displs);
    free(local_matrix);
    free(local_vector);
    free(local_result);
    if (rank == 0) {
        free(matrix);
        free(vector);
        free(result);
    }

    MPI_Finalize();
    return 0;
}