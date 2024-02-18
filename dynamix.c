#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 600
#define MAX_ITER 1000

int main(int argc, char** argv) {
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double t_start = MPI_Wtime();

    int num_tasks = WIDTH * HEIGHT;
    int tasks_per_proc = ceil(num_tasks / (double)size);

    int start_task = rank * tasks_per_proc;
    int end_task = fmin(start_task + tasks_per_proc, num_tasks);

    int mandelbrot[HEIGHT][WIDTH] = {0};
    for (int task = start_task; task < end_task; task++) {
        int x = task % WIDTH;
        int y = task / WIDTH;

        double c_re = (x - WIDTH/2.0)*4.0/WIDTH;
        double c_im = (y - HEIGHT/2.0)*4.0/WIDTH;
        double z_re = 0, z_im = 0;

        int iter;
        for (iter = 0; iter < MAX_ITER; iter++) {
            double z_re_new = z_re*z_re - z_im*z_im + c_re;
            double z_im_new = 2*z_re*z_im + c_im;
            z_re = z_re_new;
            z_im = z_im_new;

            if (z_re*z_re + z_im*z_im > 4) {
                break;
            }
        }
        mandelbrot[y][x] = iter;
    }

    // Gathering computed results to root process
    int* recvcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        int num_tasks_i = i == size - 1 ? num_tasks - i * tasks_per_proc : tasks_per_proc;
        recvcounts[i] = num_tasks_i;
        displs[i] = i * tasks_per_proc;
    }

    int* mandelbrot_all = NULL;
    if (rank == 0) {
        mandelbrot_all = malloc(num_tasks * sizeof(int));
    }
    MPI_Gatherv(&(mandelbrot[0][start_task]), end_task - start_task, MPI_INT,
                mandelbrot_all, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE* fp = fopen("mandelbrot.ppm", "wb");
        fprintf(fp, "P6 %d %d 255\n", WIDTH, HEIGHT);
        for (int i = 0; i < num_tasks; i++) {
            int color = mandelbrot_all[i] * 255 / MAX_ITER;
            fputc(color, fp);
            fputc(color, fp);
            fputc(color, fp);
        }
        fclose(fp);
        free(mandelbrot_all);

        double t_end = MPI_Wtime();
        printf("Execution time: %.2f seconds\n", t_end - t_start);
    }

    MPI_Finalize();
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time used: %f seconds\n", cpu_time_used);
    
    return 0;
}
