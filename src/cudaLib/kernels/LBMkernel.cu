#include <LBMkernel.cuh>
#include <gpuOperations.cuh>
#include <cudaHelper.cuh>
#include <cudaMathHelper.cuh>

// Kernel functions
__global__ void kernelUpdateForce(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {
        if (x >= 10 && x <= 20 && y >= 10 && y <= 20) {
            fluid->force3[3 * idx] = 1e-4;
        }
    }
}

// Kernel functions
__global__ void kernelFcopy(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {
        for (int d = 0; d < dim; d++) {
            fluid->fOldDim[dim * idx + d] = fluid->fNewDim[dim * idx + d];
        }
    }
}

// Kernel functions
__global__ void kernelForce(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {

        float Fx = fluid->force3[3 * idx];
        float Fy = fluid->force3[3 * idx + 1];
        float Fz = fluid->force3[3 * idx + 2];

        float ux = fluid->vel3[3 * idx];
        float uy = fluid->vel3[3 * idx + 1];
        float uz = fluid->vel3[3 * idx + 2];

        // Calculate force term
        for (int d = 0; d < dim; ++d) {
            float cix = (float) solver->ciX[d];
            float ciy = (float) solver->ciY[d];
            float ciz = (float) solver->ciZ[d];

            float term1 = (cix - ux) * Fx + (ciy - uy) * Fy + (ciz - uz) * Fz;
            float term2 = (cix * ux + ciy * uy + ciz * uz) * (cix * Fx + ciy * Fy + ciz * Fz);

            fluid->gridForceDim[dim * idx + d] = (1 - 0.5 * solver->relaxFreq) * solver->w[d] * (3 * term1 + 9 * term2);
        }
    }
}

// Kernel functions
__global__ void kernelEqu(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {

        float ux = fluid->vel3[3 * idx];
        float uy = fluid->vel3[3 * idx + 1];
        float uz = fluid->vel3[3 * idx + 2];

        float dens = fluid->density1[idx];

        // Calculate equal state
        for (int d = 0; d < dim; ++d) {
            float cix = solver->ciX[d];
            float ciy = solver->ciY[d];
            float ciz = solver->ciZ[d];

            float term1 = cix * ux + ciy * uy + ciz * uz;
            float term2 = ux * ux + uy * uy + uz * uz;

            fluid->equDim[dim * idx + d] = solver->w[d] * dens * (1 + csSquareInv * term1 + 0.5 * pow(csSquareInv, 2) * term1 * term1 - 0.5 * csSquareInv * term2);
        }
    }
}

__global__ void kernelColliStream(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {

        // LBM equation, includes both streaming and collision
        for (int d = 0; d < dim; ++d) {
            int nextX = (x + solver->ciX[d] + fluid->nX) % fluid->nX;
            int nextY = (y + solver->ciY[d] + fluid->nY) % fluid->nY;
            int nextZ = (z + solver->ciZ[d] + fluid->nZ) % fluid->nZ;

            // Collision
            float combinedf = 0;
            if (colliModel == BGK) {
                combinedf = (1 - relaxFreq) * fluid->fOldDim[dim * idx + d] + relaxFreq * fluid->equDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
            } else if (colliModel == MRT) {
                combinedf += fluid->fOldDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
                for (int k = 0; k < dim; k++) {
                    combinedf += solver->MinvSM[d * dim + k] * (fluid->equDim[dim * idx + k] - fluid->fOldDim[dim * idx + k]);
                }
            }
            
            // Stream (Bounce Back)
            int nextIdx = getIndex(fluid, nextX, nextY, nextZ);
            
            if (fluid->status1[nextIdx] == status_solid) {
                fluid->fNewDim[dim * idx + solver->inversed[d]] = combinedf;
            } else {
                fluid->fNewDim[dim * nextIdx + d] = combinedf;
            }
        }
    }
}

// Kernel functions
__global__ void kernelUpdateQuantity(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = getIndex(fluid, x, y, z);

    if (inRange(fluid, x, y, z) && fluid->status1[idx] == status_fluid) {

        fluid->density1[idx] = 0;
        fluid->vel3[3 * idx    ] = 0.5 * (fluid->force3[3 * idx    ]);
        fluid->vel3[3 * idx + 1] = 0.5 * (fluid->force3[3 * idx + 1]);
        fluid->vel3[3 * idx + 2] = 0.5 * (fluid->force3[3 * idx + 2]);

        for (int d = 0; d < dim; ++d) {
            fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
            fluid->vel3[3 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel3[3 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel3[3 * idx + 2] += solver->ciZ[d] * fluid->fNewDim[dim * idx + d];
        }
        
        fluid->vel3[3 * idx    ] = fluid->vel3[3 * idx    ] / fluid->density1[idx];
        fluid->vel3[3 * idx + 1] = fluid->vel3[3 * idx + 1] / fluid->density1[idx];
        fluid->vel3[3 * idx + 2] = fluid->vel3[3 * idx + 2] / fluid->density1[idx];
    }
}

// The LBM algorithm
void colliAndAdvect(Solver * gpuSolver, Fluid * gpuFluid, int t)
{
    dim3 dimGrid(ceil((float) gpuFluid->nX / (float) tileSize), 
                 ceil((float) gpuFluid->nX / (float) tileSize), 
                 ceil((float) gpuFluid->nX / (float) tileSize));
    dim3 dimBlock(tileSize, tileSize, tileSize);

    kernelUpdateForce<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid, t);
    cudaCheck(cudaDeviceSynchronize());

    kernelFcopy<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid);
    cudaCheck(cudaDeviceSynchronize());

    kernelForce<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid, t);
    cudaCheck(cudaDeviceSynchronize());

    kernelEqu<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid);
    cudaCheck(cudaDeviceSynchronize());

    kernelColliStream<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid);
    cudaCheck(cudaDeviceSynchronize());

    kernelUpdateQuantity<<<dimGrid, dimBlock>>>(gpuSolver, gpuFluid, t);
    cudaCheck(cudaDeviceSynchronize());
}