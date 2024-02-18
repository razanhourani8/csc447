#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define MASTER_RANK 0

void c_mandelbrot(int width, int height, int start_row, int end_row, double left, double right, double lower, double upper, int* output) {
    int max_iterations = 1000;
    double x, y, x_n, y_n, x_sq, y_sq;
    int iter;

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < width; j++) {
            x = left + (right - left) * j / width;
            y = lower + (upper - lower) * i / height;
            x_n = 0.0;
            y_n = 0.0;
            x_sq = 0.0;
            y_sq = 0.0;
            iter = 0;

            while (x_sq + y_sq < 4.0 && iter < max_iterations) {
                y_n = 2 * x_n * y + y;
                x_new = x_sq - y_sq + x;
                x_sq = x_n * x_n;
                y_sq = y_n * y_n;
                iter++;
            }

            output[i * width + j] = iter;
        }
    }
}

int main(int argc, char** argv) {
    clock_t start, end;
    double cpu_time_used;

    int width = 800, height = 800;
    double left = -2.0, right = 1.0, lower = -1.5, upper = 1.5;
    int max_iterations = 1000;

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = height / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;
    if (rank == size - 1) {
        end_row = height;
    }
    int row_count = end_row - start_row;

    int* local_output = (int*)malloc(row_count * width * sizeof(int));
    c_mandelbrot(width, height, start_row, end_row, left, right, lower, upper, local_output);

    int* global_output = NULL;
    if (rank == MASTER_RANK) {
        global_output = (int*)malloc(width * height * sizeof(int));
    }

    MPI_Gather(local_output, row_count * width, MPI_INT, global_output, row_count * width, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

    if (rank == MASTER_RANK) {
        FILE* file = fopen("mandelbrot.pgm", "wb");
        fprintf(file, "P2\n%d %d\n%d\n", width, height, max_iter - 1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                fprintf(file, "%d ", global_output[i * width + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        free(global_output);
    }

    free(local_output);
    MPI_Finalize();

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    return 0;
}
