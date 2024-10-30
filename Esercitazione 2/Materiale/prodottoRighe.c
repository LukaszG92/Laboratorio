#include <stdio.h>
#include <stdlib.h>
#include <mpi.h> 

#define ROWS 40000
#define COLS 40000

int main(int argc, char **argv) {
    int nproc, rank, local_ROWS, i, j;
    double *matrix, *vector, *local_matrix,*local_result, *result;
    double start_time, end_time, max_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    local_ROWS = ROWS / nproc;
    int remainder = ROWS % nproc;
    if (rank < remainder) local_ROWS++;

    vector = malloc(sizeof(double)*COLS);   
    local_matrix = malloc(local_ROWS * COLS * sizeof(double));
    local_result = malloc(local_ROWS * sizeof(double));

    if(rank == 0) { 
        matrix = malloc(ROWS * COLS * sizeof(double));
        result =  malloc(sizeof(double)*ROWS); 

        for (i = 0; i < ROWS; i++) { 
            for(j = 0; j < COLS; j++) {
                    matrix[i*COLS+j]=i;
            }
        }

        for (j = 0; j < COLS; j++) {
            vector[j]=1; 
        }
        if(ROWS <= 10 && COLS <= 10) {
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                printf("%f ", matrix[i * ROWS + j]);
            }
        printf("\n");
        }
    }
    }
        
    MPI_Bcast(&vector[0],COLS,MPI_DOUBLE,0,MPI_COMM_WORLD);            

    int *sendcounts = (int*)malloc(nproc * sizeof(int));
    int *displs = (int*)malloc(nproc * sizeof(int));
    int *recvcounts = (int*)malloc(nproc * sizeof(int));
    int *rdispls = (int*)malloc(nproc * sizeof(int));
    
    int disp = 0;
    int rdisp = 0;
    for (int i = 0; i < nproc; i++) {
        int rows_for_proc = ROWS / nproc;
        if (i < remainder) rows_for_proc++;
        
        sendcounts[i] = rows_for_proc * ROWS;
        displs[i] = disp * ROWS;
        disp += rows_for_proc;

        recvcounts[i] = rows_for_proc;
        rdispls[i] = rdisp;
        rdisp += rows_for_proc;
    }

    MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE,
        local_matrix, sendcounts[rank], MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    for(i = 0; i < local_ROWS; i++) {   
        local_result[i] = 0;
        for(j = 0; j < COLS; j++) {
            local_result[i] += local_matrix[i * COLS + j] * vector[j];
        }
    }    
        
    MPI_Gatherv(local_result, local_ROWS, MPI_DOUBLE,
                result, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime() - start_time;

    MPI_Reduce(&end_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nRisultati:\n");
        printf("Dimensioni matrice: %d x %d\n", ROWS, COLS);
        printf("Numero processori: %d\n", nproc);
        printf("Results: %f, %f, %f, %f, %f.\n", result[1], result[2], result[3], result[4], result[5]);
        printf("Tempo di esecuzione: %f secondi\n", max_time);
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

    MPI_Finalize ();
    return 0;  
}