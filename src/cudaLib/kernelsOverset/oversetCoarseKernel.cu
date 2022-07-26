#include <oversetFineKernel.cuh>
#include <cudaHelper.cuh>
#include <cudaMathHelper.cuh>
#include <assert.h>
#include <iostream>

// ---------- Kernel functions ---------- 
__global__ void kernelUpdateForceCoarse(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {        
        if (x >= 10 && x <= 50 && y >= fluid->nY / 2.0f - 20 && y <= fluid->nY / 2.0f + 20) {
            fluid->force2[2 * idx] = 1e-5;
            fluid->force2[2 * idx + 1] = 0;
            //printf("%f", fluid->force2[2 * idx]);
            //printf("%d ", dim);
            //fluid->vel2[2 * idx] = 0.05;

        }
        else if (x >= 750 && x <= 790 && y >= fluid->nY / 2.0f - 20 && y <= fluid->nY / 2.0f + 20) {
            fluid->force2[2 * idx] = -1e-5;
            fluid->force2[2 * idx + 1] = 0;
            //printf("%f", fluid->force2[2 * idx]);
            //printf("%d ", dim);
            //fluid->vel2[2 * idx] = 0.05;

        }
        else {
            fluid->force2[2 * idx] = 0;
            fluid->force2[2 * idx + 1] = 0;
        }
        

        // if (scale == 1) {
        //     if (x >= 2 && x <= 20 && y >= fluid->nY / 2.0f - 5 && y <= fluid->nY / 2.0f + 5) {
        //         fluid->force2[2 * idx] = 3e-4;
        //     }
        // } else if (scale == 2) {
        //     if (x >= 5 && x <= 10 && y >= fluid->nY / 2.0f - 5 && y <= fluid->nY / 2.0f + 5) {
        //         fluid->force2[2 * idx] = 3e-4;
        //     }
        // } else if (scale == 10) {
        //     if (x >= 2 && x <= 5 && y >= fluid->nY / 2.0f - 2 && y <= fluid->nY / 2.0f + 2) {
        //         fluid->force2[2 * idx] = 3e-4;
        //     }
        // }
    }
}

__global__ void kernelFcopyCoarse(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y)) {

        // Store old values
        fluid->densityOld1[idx] = fluid->density1[idx];
        for (int i = 0; i < 2; i++) {
            fluid->velOld2[2 * idx + i] = fluid->vel2[2 * idx + i];
        }
        for (int d = 0; d < dim; d++) {
            fluid->fOldDim[dim * idx + d] = fluid->fNewDim[dim * idx + d];
            fluid->fLastDim[dim * idx + d] = fluid->fNewDim[dim * idx + d];
        }

        // Set all streamed label to false
        for (int d = 0; d < dim; d++) {
            fluid->streamedDim[dim * idx + d] = false;
        }
    }
}

__global__ void kernelForceCoarse(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {

        float Fx = fluid->force2[2 * idx];
        float Fy = fluid->force2[2 * idx + 1];

        float ux = fluid->vel2[2 * idx];
        float uy = fluid->vel2[2 * idx + 1];

        // Calculate force term
        for (int d = 0; d < dim; ++d) {
            float cix = (float) solver->ciX[d];
            float ciy = (float) solver->ciY[d];

            float term1 = (cix - ux) * Fx + (ciy - uy) * Fy;
            float term2 = (cix * ux + ciy * uy) * (cix * Fx + ciy * Fy);

            fluid->gridForceDim[dim * idx + d] = (1 - 0.5 * solver->relaxFreq) * solver->w[d] * (3 * term1 + 9 * term2);
        }
    }
}

__global__ void kernelEquCoarse(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    float csSquareInv = 3;

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {

        float ux = fluid->vel2[2 * idx];
        float uy = fluid->vel2[2 * idx + 1];

        float dens = fluid->density1[idx];

        // Calculate equal state
        for (int d = 0; d < dim; ++d) {
            float cix = solver->ciX[d];
            float ciy = solver->ciY[d];

            float term1 = cix * ux + ciy * uy;
            float term2 = ux * ux + uy * uy;

            fluid->equDim[dim * idx + d] = solver->w[d] * dens * (1 + csSquareInv * term1 + 0.5 * pow(csSquareInv, 2) * term1 * term1 - 0.5 * csSquareInv * term2);
        }
    }
}

__global__ void kernelColliStreamCoarse(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {

        // LBM equation, includes both streaming and collision
        for (int d = 0; d < dim; ++d) {
            int nextX = (x + solver->ciX[d] + fluid->nX) % fluid->nX;
            int nextY = (y + solver->ciY[d] + fluid->nY) % fluid->nY;
            int nextIdx = getIndex(fluid, nextX, nextY);

            if (fluid->status1[nextIdx] != status_invalid) {
                // Collision
                float combinedf = 0;
                if (colliModel == BGK) {
                    combinedf = (1 - solver->relaxFreq) * fluid->fOldDim[dim * idx + d] + solver->relaxFreq * fluid->equDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
                } else if (colliModel == MRT) {
                    //printf("0");
                    combinedf += fluid->fOldDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
                    for (int k = 0; k < dim; k++) {
                        combinedf += solver->MinvSM[d * dim + k] * (fluid->equDim[dim * idx + k] - fluid->fOldDim[dim * idx + k]);
                    }
                }
                else {
                    printf("111");
                }
                
                // Stream (Bounce Back)
                if (fluid->status1[nextIdx] == status_solid) {
                    fluid->fNewDim[dim * idx + solver->inversed[d]] = combinedf;
                    fluid->streamedDim[dim * idx + solver->inversed[d]] = true;
                } else {
                    fluid->fNewDim[dim * nextIdx + d] = combinedf;
                    fluid->streamedDim[dim * nextIdx + d] = true;
                }
            }
        }
    }
}

__global__ void kernelUpdateQuantityCoarse(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        // Initialize
        fluid->density1[idx] = 0;
        fluid->vel2[2 * idx    ] = 0.5 * (fluid->force2[2 * idx    ]);
        fluid->vel2[2 * idx + 1] = 0.5 * (fluid->force2[2 * idx + 1]);
        // Density and Velocity
        for (int d = 0; d < dim; ++d) {
            fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        fluid->vel2[2 * idx    ] = fluid->vel2[2 * idx    ] / fluid->density1[idx];
        fluid->vel2[2 * idx + 1] = fluid->vel2[2 * idx + 1] / fluid->density1[idx];

        /*if (t % 100 == 0 && x == 7 && y == 80) {
            printf("%f ", fluid->vel2[2 * idx]);
        }*/
    }
}

// ---------- Main function ---------- 
void colliAndAdvectCoarse(Solver ** gpuSolverList, Fluid ** gpuFluidList, Solver ** solverList, Fluid ** fluidList, int t, int numFineFluids, int interpolateType)
{
    dim3 dimGrid(ceil((float) fluidList[0]->nX / (float) tileSize), 
                 ceil((float) fluidList[0]->nY / (float) tileSize), 
                 1);
    dim3 dimBlock(tileSize, tileSize, 1);
    //std::cout << 1;
    /*initInvalidRegionPosCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0], t);
    cudaCheck(cudaDeviceSynchronize());*/

    //std::cout << 2;
    kernelUpdateForceCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0], t);
    cudaCheck(cudaDeviceSynchronize());
    //std::cout << 3;
    kernelFcopyCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0]);
    cudaCheck(cudaDeviceSynchronize());
    //std::cout << 4;
    kernelForceCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0], t);
    cudaCheck(cudaDeviceSynchronize());
    //std::cout << 5;
    kernelEquCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0]);
    cudaCheck(cudaDeviceSynchronize());
    //std::cout << 6;
    kernelColliStreamCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0]);
    cudaCheck(cudaDeviceSynchronize());

    kernelUpdateQuantityCoarse << <dimGrid, dimBlock >> > (gpuSolverList[0], gpuFluidList[0], t);
    cudaCheck(cudaDeviceSynchronize());

    // Bilinear interpolation
    /*if (interpolateType == 0) {
        for (int i = 1; i < numFineFluids + 1; i++) {
            kernelInterpolateCoarse<<<dimGrid, dimBlock>>>(gpuSolverList[0], gpuFluidList[0], gpuSolverList[i], gpuFluidList[i], t);
            cudaCheck(cudaDeviceSynchronize());
        }
        
    } */
    /*else {
        printf("Failure: No interpolation type %d (CoarseGrid)\n", interpolateType);
    }*/
}
