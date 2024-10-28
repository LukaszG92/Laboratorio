#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARRAY_SIZE 100000000

int main(int argc, char **argv) {
    int rank, nproc, local_size, p, tmp;
    int *data, *local_data, *powers;
    int steps = 0;
    int local_sum = 0;
    double start_time, end_time, max_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    
    // Calcolo della dimensione locale per ogni processo
    local_size = ARRAY_SIZE / nproc;
    if (rank < (ARRAY_SIZE % nproc)) {
        local_size++;
    }
    
    // Allocazione del buffer locale
    local_data = (int*)malloc(local_size * sizeof(int));
    
    // Il processo root inizializza il vettore
    if (rank == 0) {
        data = (int*)malloc(ARRAY_SIZE * sizeof(int));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = 1;
        }
    }
    
    // Utilizzo di MPI_Scatterv per gestire dimensioni non uniformi
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

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Distribuzione del vettore usando MPI_Scatterv
    MPI_Scatterv(data, sendcounts, displs, MPI_INT, 
                 local_data, local_size, MPI_INT, 
                 0, MPI_COMM_WORLD);

    
    // Calcolo della somma locale
    for (int i = 0; i < local_size; i++) {
        local_sum += local_data[i];
    }
    
    // Calcolo numero di passi per la riduzione
    p = nproc;
    while (p != 1) {
        p = p >> 1;
        steps++;
    }
    
    powers = (int*)calloc(steps + 1, sizeof(int));
    for (int i = 0; i <= steps; i++) {
        powers[i] = 1 << i;
    }
    
    // Riduzione della somma
    for (int i = 0; i < steps; i++) {
        int remainder = rank % powers[i + 1];
        
        if (remainder == powers[i]) {
            int send_to = rank - powers[i];
            MPI_Send(&local_sum, 1, MPI_INT, send_to, send_to, MPI_COMM_WORLD);
        }
        else if (remainder == 0) {
            int receive_from = rank + powers[i];
            MPI_Recv(&tmp, 1, MPI_INT, receive_from, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
        
        free(sendcounts);
        free(displs);
        free(data);
    }
    
    free(local_data);
    free(powers);
    
    MPI_Finalize();
    return 0;
}