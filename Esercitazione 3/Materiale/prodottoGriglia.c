#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define ROWS 40000
#define COLS 40000

// Function to find the best grid dimensions
void get_grid_dims(int size, int *p, int *q) {
    // Find factors of size
    int sqrt_size = (int)sqrt(size);
    *p = sqrt_size;
    while (*p >= 1) {
        if (size % *p == 0) {
            *q = size / *p;
            return;
        }
        (*p)--;
    }
    // Fallback to 1 x size grid if no better factors found
    *p = 1;
    *q = size;
}

int main(int argc, char **argv) {
    int rank, size;
    double *matrix = NULL, *vector = NULL, *result = NULL;
    double *local_matrix, *local_vector, *local_result;
    double start_time, end_time;
    MPI_Comm grid_comm, row_comm, col_comm;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate grid dimensions
    int p, q;  // p rows, q columns in the grid
    get_grid_dims(size, &p, &q);
    
    // Create 2D Cartesian grid
    int dims[2] = {p, q};
    int periods[2] = {0, 0};  // Non-periodic
    int coords[2];            // Will store coordinates of this process
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    
    // Create row and column communicators
    int remain_dims[2];
    
    // Row communicator
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(grid_comm, remain_dims, &row_comm);
    
    // Column communicator
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(grid_comm, remain_dims, &col_comm);
    
    // Calculate local dimensions with potential remainders
    int local_rows = ROWS / p + (coords[0] < ROWS % p ? 1 : 0);
    int local_cols = COLS / q + (coords[1] < COLS % q ? 1 : 0);
    
    // Calculate displacements
    int row_disp = (ROWS / p) * coords[0] + (coords[0] < ROWS % p ? 1 : 0);
    int col_disp = (COLS / q) * coords[1] + (coords[1] < COLS % q ? 1 : 0);
    
    // Allocate memory for local data
    local_matrix = (double*)malloc(local_rows * local_cols * sizeof(double));
    local_vector = (double*)malloc(local_cols * sizeof(double));
    local_result = (double*)malloc(local_rows * sizeof(double));
    
    // Root process initializes the data
    if (rank == 0) {
        matrix = (double*)malloc(ROWS * COLS * sizeof(double));
        vector = (double*)malloc(COLS * sizeof(double));
        result = (double*)malloc(ROWS * sizeof(double));
        
        // Initialize matrix
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                matrix[i * COLS + j] = i;
            }
        }
        
        // Initialize vector
        for (int i = 0; i < COLS; i++) {
            vector[i] = 1.0;
        }
        
        // Print matrix for small dimensions
        if (ROWS <= 10 && COLS <= 10) {
            printf("Initial Matrix:\n");
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    printf("%f ", matrix[i * COLS + j]);
                }
                printf("\n");
            }

            printf("Vector:\n");
            for(int i = 0; i < COLS; i++) {
                printf("%f ", vector[i]);
            }
            printf("\n");
        }
    }

    // Calculate send counts and displacements for matrix distribution
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < q; j++) {
                int proc_rank;
                int proc_coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, proc_coords, &proc_rank);
                
                int proc_rows = ROWS / p + (i < ROWS % p ? 1 : 0);
                int proc_cols = COLS / q + (j < COLS % q ? 1 : 0);
                
                sendcounts[proc_rank] = proc_rows * proc_cols;
                
                int row_offset = (ROWS / p) * i + (i < ROWS % p ? 1 : 0);
                int col_offset = (COLS / q) * j + (j < COLS % q ? 1 : 0);
                displs[proc_rank] = row_offset * COLS + col_offset;
            }
        }
    }
    
    // Create and commit vector datatype for non-contiguous matrix blocks
    MPI_Datatype block_type;
    MPI_Type_vector(local_rows, local_cols, COLS, MPI_DOUBLE, &block_type);
    MPI_Type_commit(&block_type);
    
    // Create a resized type to handle proper displacement in bytes
    MPI_Datatype resized_block_type;
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block_type);
    MPI_Type_commit(&resized_block_type);
    
    // Distribute matrix blocks
    if (rank == 0) {
        // Send to all other processes (including self)
        for (int i = 0; i < size; i++) {
            int dest_coords[2];
            MPI_Cart_coords(grid_comm, i, 2, dest_coords);
            
            int dest_rows = ROWS / p + (dest_coords[0] < ROWS % p ? 1 : 0);
            int dest_cols = COLS / q + (dest_coords[1] < COLS % q ? 1 : 0);
            
            if (i == 0) {
                // Copy local portion for rank 0
                for (int r = 0; r < dest_rows; r++) {
                    for (int c = 0; c < dest_cols; c++) {
                        local_matrix[r * dest_cols + c] = 
                            matrix[(row_disp + r) * COLS + (col_disp + c)];
                    }
                }
            } else {
                MPI_Send(&matrix[displs[i]], 1, resized_block_type, 
                        i, 0, grid_comm);
            }
        }
    } else {
        MPI_Recv(local_matrix, local_rows * local_cols, MPI_DOUBLE, 
                 0, 0, grid_comm, MPI_STATUS_IGNORE);
    }
    
    // Distribute vector
    int *vec_recvcounts = (int*)malloc(q * sizeof(int));
    int *vec_displs = (int*)malloc(q * sizeof(int));
    
    if (coords[0] == 0) {
        // Calculate vector distribution parameters
        for (int i = 0; i < q; i++) {
            vec_recvcounts[i] = COLS / q + (i < COLS % q ? 1 : 0);
            vec_displs[i] = (i > 0) ? vec_displs[i-1] + vec_recvcounts[i-1] : 0;
        }
        
        // Scatter vector among processes in first row
        MPI_Scatterv(vector, vec_recvcounts, vec_displs, MPI_DOUBLE,
                    local_vector, vec_recvcounts[coords[1]], MPI_DOUBLE,
                    0, row_comm);
    }
    
    // Broadcast local vector portions down the columns
    MPI_Bcast(local_vector, local_cols, MPI_DOUBLE, 0, col_comm);
    
    // Synchronize and start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Compute local matrix-vector product
    for (int i = 0; i < local_rows; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < local_cols; j++) {
            local_result[i] += local_matrix[i * local_cols + j] * local_vector[j];
        }
    }
    
    // Reduce along rows to get partial sums
    double* row_sums = NULL;
    if (coords[1] == 0) {
        row_sums = (double*)malloc(local_rows * sizeof(double));
    }
    MPI_Reduce(local_result, row_sums, local_rows, MPI_DOUBLE,
               MPI_SUM, 0, row_comm);
    
    // Gather final results to root
    if (coords[1] == 0) {
        // Calculate gathering parameters for rows
        int *row_recvcounts = NULL;
        int *row_displs = NULL;
        
        if (rank == 0) {
            row_recvcounts = (int*)malloc(p * sizeof(int));
            row_displs = (int*)malloc(p * sizeof(int));
            
            for (int i = 0; i < p; i++) {
                row_recvcounts[i] = ROWS / p + (i < ROWS % p ? 1 : 0);
                row_displs[i] = (i > 0) ? row_displs[i-1] + row_recvcounts[i-1] : 0;
            }
        }
        
        MPI_Gatherv(row_sums, local_rows, MPI_DOUBLE,
                   result, row_recvcounts, row_displs, MPI_DOUBLE,
                   0, col_comm);
                   
        if (rank == 0) {
            free(row_recvcounts);
            free(row_displs);
        }
    }
    
    // Synchronize and stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Print results
    if (rank == 0) {
        printf("\nResults:\n");
        printf("Matrix dimensions: %d x %d\n", ROWS, COLS);
        printf("Grid dimensions: %d x %d\n", p, q);
        printf("Number of processors: %d\n", size);
        printf("First 5 elements of result: %f %f %f %f %f\n",
               result[0], result[1], result[2], result[3], result[4]);
        printf("Execution time: %f seconds\n", end_time - start_time);
    }
    
    // Clean up
    free(local_matrix);
    free(local_vector);
    free(local_result);
    free(vec_recvcounts);
    free(vec_displs);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
        free(matrix);
        free(vector);
        free(result);
    }
    if (coords[1] == 0) {
        free(row_sums);
    }
    
    MPI_Type_free(&block_type);
    MPI_Type_free(&resized_block_type);
    MPI_Comm_free(&grid_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}