#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <errno.h>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <iostream>

// TASK: T1
// Include the cooperative groups library
// BEGIN: T1
;
#include <cooperative_groups.h>

namespace cu = cooperative_groups;

// END: T1


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)


// Option to change numerical precision
typedef int64_t int_t;
typedef double real_t;

// TASK: T1b
// Variables needed for implementation
// BEGIN: T1b


// Simulation parameters: size, step count, and how often to save the state
int_t
    N = 128,
    M = 128,
    max_iteration = 1000000,
    snapshot_freq = 1000;

// Wave equation parameters, time step is derived from the space step
const real_t
    c  = 1.0,
    dx = 1.0,
    dy = 1.0;
real_t
    dt;

// Buffers for three time steps, indexed with 2 ghost points for the boundary
real_t
    *d_buffers[3] = { NULL, NULL, NULL }; // device buffers
real_t
    *h_buffer = NULL; //host buffer

#define U_prv(i,j) d_buffers[0][((i)+1)*(N+2)+(j)+1]
#define U(i,j)     d_buffers[1][((i)+1)*(N+2)+(j)+1]
#define U_nxt(i,j) d_buffers[2][((i)+1)*(N+2)+(j)+1]
// END: T1b

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// Rotate the time step buffers.
void move_buffer_window ( void )
{
    real_t *temp = d_buffers[0];
    d_buffers[0] = d_buffers[1];
    d_buffers[1] = d_buffers[2];
    d_buffers[2] = temp;
}


// Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];

    // Ensure output directory exists (ignore error if it already exists)
    if (mkdir("data", 0755) != 0 && errno != EEXIST) {
        perror("mkdir data");
        exit(EXIT_FAILURE);
    }

    snprintf(filename, sizeof(filename), "data/%05" PRId64 ".dat", step);

    FILE *out = fopen(filename, "wb");
    if (out == NULL) {
        perror("fopen output file");
        fprintf(stderr, "Failed to open '%s' for writing.\n", filename);
        exit(EXIT_FAILURE);
    }

    for ( int_t i = 0; i < M; ++i ) {
        size_t written = fwrite ( &U(i,0), sizeof(real_t), (size_t)N, out );
        if ( written != (size_t)N ) {
            perror("fwrite");
            fclose(out);
            exit(EXIT_FAILURE);
        }
    }

    if ( fclose(out) != 0 ) {
        perror("fclose");
        exit(EXIT_FAILURE);
    }
}


// TASK: T4
// Get rid of all the memory allocations
void domain_finalize ( void )
{
// BEGIN: T4
    cudaFree ( d_buffers[0] );
    cudaFree ( d_buffers[1] );
    cudaFree ( d_buffers[2] );
    free (h_buffer);
// END: T4
}


// TASK: T6
// Neumann (reflective) boundary condition
// BEGIN: T6
__device__ void boundary_condition ( real_t *cur, int_t M, int_t N, int_t i, int_t j ){
    real_t *d_buffers[3] = { NULL, cur, NULL };

    if ( i >= M || j >= N ) return;

      if ( j == 0 )   U(i,-1) = U(i,1);
      if ( j == N-1 ) U(i,N)  = U(i,N-2);
      if ( i == 0 )   U(-1,j) = U(1,j);
      if ( i == M-1 ) U(M,j)  = U(M-2,j);
  }
// END: T6


// TASK: T5
// Integration formula
// BEGIN: T5
void __global__ time_step(real_t *prv, real_t *cur, real_t *nxt, int_t M, int_t N, real_t dt){
    auto grid = cu::this_grid();
    real_t *d_buffers[3] = { prv, cur, nxt };
    int_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int_t i = blockIdx.y * blockDim.y + threadIdx.y;
    boundary_condition(cur, M, N, i, j);
    grid.sync();

    if (i < M && j < N) {
        U_nxt(i,j) = -U_prv(i,j) + 2.0*U(i,j)
                 + (dt*dt*c*c)/(dx*dy) * (
                    U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1)-4.0*U(i,j)
                 );
    }
}
// END: T5


// TASK: T7
// Main time integration.
void simulate( void )
{
// BEGIN: T7
    dim3 block ( 16, 16 );
    dim3 grid  ( (N + block.x - 1) / block.x,
                 (M + block.y - 1) / block.y );
    // Go through each time step
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            cudaMemcpy ( h_buffer, d_buffers[1],(M+2)*(N+2)*sizeof(real_t),cudaMemcpyDeviceToHost );
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        void *args[] = { &d_buffers[0], &d_buffers[1], &d_buffers[2], &M, &N, &dt };
        cudaLaunchCooperativeKernel ( (void*)time_step, grid, block, args );

        // Rotate the time step buffers
        move_buffer_window();
    }
// END: T7
}


// TASK: T8
// GPU occupancy
void occupancy( void )
{
// BEGIN: T8
    ;
    cudaDeviceProp prop;
    cudaGetDeviceProperties ( &prop, 0 );

    int threads_per_block = 16 * 16;   // must match the block size in simulate()

    // How many blocks of time_step fit on one SM at the same time
    int active_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor (
                        &active_blocks_per_sm, time_step, threads_per_block, 0 );

    int warps_per_block  = (threads_per_block + prop.warpSize - 1) / prop.warpSize;
    int active_warps     = active_blocks_per_sm * warps_per_block;
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    double occ = (double)active_warps / max_warps_per_sm;

    printf ( "Grid size set to: %" PRId64 " x %" PRId64 "\n", (N + 15) / 16, (M + 15) / 16 );
    printf ( "Block size set to: 16 x 16\n" );
    printf ( "Active blocks per SM: %d\n", active_blocks_per_sm );
    printf ( "Theoretical occupancy: %.2f%%\n", occ * 100.0 );
// END: T8
}


// TASK: T2
// Make sure at least one CUDA-capable device exists
static bool init_cuda()
{
// BEGIN: T2

    int device_count = 0;
    auto err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    std::cout << "Device count " << device_count << std::endl;

    // print properties

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "\tName: " << prop.name << std::endl;
        std::cout << "\tCompute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "\tMultiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "\tWarp size: " << prop.warpSize << std::endl;
        std::cout << "\tGlobal memory: " << prop.totalGlobalMem << std::endl;
        std::cout << "\tPer block shared memory: " << prop.sharedMemPerBlock << std::endl;
        std::cout << "\tPer Block registers: " << prop.regsPerBlock << std::endl;
        std::cout << std::endl;
    }


    // set device
    std::cout << "Setting device to 0" << std::endl;
    cudaSetDevice(0);


    return true;
// END: T2
}


// TASK: T3
// Set up our three buffers, and fill two with an initial perturbation
// Function to determine occupancy and optimal configuration


void domain_initialize ( void )
{
// BEGIN: T3
    bool locate_cuda = init_cuda();
    if (!locate_cuda)
    {
        exit( EXIT_FAILURE );
    }

    h_buffer = (real_t *) malloc ( (M+2)*(N+2)*sizeof(real_t) );
    d_buffers[0] = d_buffers[1] = h_buffer;

    for ( int_t i=0; i<M; i++ )
    {
        for ( int_t j=0; j<N; j++ )
        {
            // Calculate delta (radial distance) adjusted for M x N grid
            real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
                                ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
            U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
        }
    }

    real_t *tmp_bufs[3] = {NULL, NULL, NULL};

    cudaMalloc(&tmp_bufs[0], (M+2)*(N+2)*sizeof(real_t));
    cudaMalloc(&tmp_bufs[1], (M+2)*(N+2)*sizeof(real_t));
    cudaMalloc(&tmp_bufs[2], (M+2)*(N+2)*sizeof(real_t));

    cudaMemcpy ( tmp_bufs[0], h_buffer, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice );
    cudaMemcpy ( tmp_bufs[1], h_buffer, (M+2)*(N+2)*sizeof(real_t), cudaMemcpyHostToDevice );
    cudaMemset ( tmp_bufs[2], 0, (M+2)*(N+2)*sizeof(real_t) );

    d_buffers[0] = tmp_bufs[0];
    d_buffers[1] = tmp_bufs[1];
    d_buffers[2] = tmp_bufs[2];

    // Set the time step for 2D case
    dt = dx*dy / (c * sqrt (dx*dx+dy*dy));
}


int main ( void )
{
    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulate();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    occupancy();

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}
