#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_PES 8
#define DATA_SIZE 100  

int main(int argc, char *argv[]) {
    int rank, size;
    int *send_data, *recv_data;
    int next_pe, prev_pe;
    MPI_Status status;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Verify we have exactly 8 PEs
    if (size != NUM_PES) {
        if (rank == 0) {
            printf("Error: This program requires exactly %d processes, got %d\n", NUM_PES, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Allocate memory for send and receive buffers
    send_data = (int*)malloc(DATA_SIZE * sizeof(int));
    recv_data = (int*)malloc(DATA_SIZE * sizeof(int));
    
    // Initialize send data - each PE creates unique data
    for (int i = 0; i < DATA_SIZE; i++) {
        send_data[i] = rank * 1000 + i;  
    }
    
    // Calculate circular neighbors
    next_pe = (rank + 1) % NUM_PES;  // Next PE (circular: PE 7 -> PE 0)
    prev_pe = (rank - 1 + NUM_PES) % NUM_PES;  // Previous PE (circular: PE 0 -> PE 7)
    
    printf("PE %d: Initialized with data range [%d-%d]\n", 
           rank, send_data[0], send_data[DATA_SIZE-1]);
    
    // Synchronize before starting communication
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Option 1: Shift "up" (each PE sends to next PE)
    printf("PE %d: Starting UPWARD shift - sending to PE %d\n", rank, next_pe);
    
    MPI_Sendrecv(send_data, DATA_SIZE, MPI_INT, next_pe, 0,
                 recv_data, DATA_SIZE, MPI_INT, prev_pe, 0,
                 MPI_COMM_WORLD, &status);
    
    printf("PE %d: Received data from PE %d, range [%d-%d]\n", 
           rank, prev_pe, recv_data[0], recv_data[DATA_SIZE-1]);
    
    // Synchronize before next shift
    MPI_Barrier(MPI_COMM_WORLD);
    
   
    printf("PE %d: Starting DOWNWARD shift - sending to PE %d\n", rank, prev_pe);
    
    // Copy received data back to send buffer for next operation
    for (int i = 0; i < DATA_SIZE; i++) {
        send_data[i] = recv_data[i];
    }
    
    // Shift in opposite direction
    MPI_Sendrecv(send_data, DATA_SIZE, MPI_INT, prev_pe, 1,
                 recv_data, DATA_SIZE, MPI_INT, next_pe, 1,
                 MPI_COMM_WORLD, &status);
    
    printf("PE %d: After downward shift, received data from PE %d, range [%d-%d]\n", 
           rank, next_pe, recv_data[0], recv_data[DATA_SIZE-1]);
    
    // Demonstrate the circular nature
    if (rank == 0) {
        printf("\nCircular shift demonstration completed:\n");
        printf("- PE 0 neighbors: prev=7, next=1\n");
        printf("- PE 7 neighbors: prev=6, next=0\n");
        printf("- Other PEs have sequential neighbors\n");
    }
    
    // Clean up
    free(send_data);
    free(recv_data);
    
    MPI_Finalize();
    return 0;
}