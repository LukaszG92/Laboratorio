#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 200000000

int main(int argc, char **argv) {
    int rank, nproc, local_size, p, tmp;
    int *data, *local_data, *powers;
    int steps = 0;
	int local_sum = 0;
    double start_time, end_time, max_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    local_size = ARRAY_SIZE / nproc;
    if (rank < (ARRAY_SIZE % nproc)) {
        local_size++;
    }
    
    local_data = (int*)malloc(local_size * sizeof(int));
    
    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = 1;
        }
    }

        p = nproc;
    while (p != 1) {
        p = p >> 1;
        steps++;
    }
    
    powers = (int*)calloc(steps + 1, sizeof(int));
    for (int i = 0; i <= steps; i++) {
        powers[i] = 1 << i;
    }
	

    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(nproc * sizeof(int));
        displs = (int*)malloc(nproc * sizeof(int));
        
        // Calcolo sendcounts e displs per ogni processo
        displs[0] = 0;
        for (int i = 0; i < nproc; i++) {
            sendcounts[i] = ARRAY_SIZE / nproc;
            if (i < (ARRAY_SIZE % nproc)) {
                sendcounts[i]++;
            }
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }
    }

    MPI_Scatterv(data, sendcounts, displs, MPI_INT, 
                 local_data, local_size, MPI_INT, 
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    for (int i = 0; i < local_size; i++) {
        local_sum += local_data[i];
    }

	
    for (int i = 0; i < steps; i++) {
        int remainder = rank % powers[i + 1];
        
        if (remainder < powers[i]) {
            int target = rank + powers[i];
            MPI_Send(&local_sum, 1, MPI_INT, target, target, MPI_COMM_WORLD);
            MPI_Recv(&tmp, 1, MPI_INT, target, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_sum += tmp;
        }
        else {
            int target = rank - powers[i];
            MPI_Recv(&tmp, 1, MPI_INT, target, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_sum, 1, MPI_INT, target, target, MPI_COMM_WORLD);
            local_sum += tmp;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime() - start_time;
    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Number of processors used: %d\n", nproc);
        printf("Number of reduction steps: %d\n", steps);
        printf("Final sum: %d\n", local_sum);
        printf("Computation time: %f seconds\n", max_time);
    }
    
    free(local_data);
    free(powers);
    if (rank == 0) {
        free(data);
    }
    
    MPI_Finalize();
    return 0;
}