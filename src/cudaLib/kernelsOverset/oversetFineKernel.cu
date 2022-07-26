#include <oversetFineKernel.cuh>
#include <cudaHelper.cuh>
#include <cudaMathHelper.cuh>
#include <iostream>
#include <assert.h>

using namespace std;

// ---------- Kernel functions ---------- 
__global__ void kernelUpdateForceFine(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        fluid->force2[2 * idx] = 0;
        fluid->force2[2 * idx + 1] = 0;

        if (fluid->scale == 1) {
            if (x >= 40 && x <= 50 && y >= fluid->nY / 2.0f - 5 && y <= fluid->nY / 2.0f + 5) {
                fluid->force2[2 * idx] = 8e-4;
            }
        } else if (fluid->scale == 10) {
            if (x >= 80 && x <= 100 && y >= fluid->nY / 2.0f - 2 && y <= fluid->nY / 2.0f + 2) { 
                fluid->force2[2 * idx] = 1e-5;
            }
        } else if (fluid->scale == 50) {
            if (x >= fluid->nX / 2 - 50 && x <= fluid->nX / 2 + 50 && y >= fluid->nY / 2 - 20 && y <= fluid->nY / 2 + 20) { 
                fluid->force2[2 * idx] = 3e-4;
            }
        } 
    }
}
  
__global__ void kernelFcopyFine(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y)) {
        for (int d = 0; d < dim; d++) {
            fluid->fOldDim[dim * idx + d] = fluid->fNewDim[dim * idx + d];
            fluid->streamedDim[dim * idx + d] = true;
        }
    }
}

__global__ void kernelForceFine(Solver * solver, Fluid * fluid, int t)
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

__global__ void kernelEquFine(Solver * solver, Fluid * fluid)
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

__global__ void kernelColliStreamFine(Solver * solver, Fluid * fluid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        for (int d = 0; d < dim; ++d) {
            int nextX = x + solver->ciX[d];
            int nextY = y + solver->ciY[d];
            int nextIdx = getIndex(fluid, nextX, nextY);

            if (inRange(fluid, nextX, nextY)) {
                // Collision
                float combinedf = 0;
                if (colliModel == BGK) {
                    combinedf = (1 - solver->relaxFreq) * fluid->fOldDim[dim * idx + d] + solver->relaxFreq * fluid->equDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
                } else if (colliModel == MRT) {
                    combinedf += fluid->fOldDim[dim * idx + d] + fluid->gridForceDim[dim * idx + d];
                    for (int k = 0; k < dim; k++) {
                        combinedf += solver->MinvSM[d * dim + k] * (fluid->equDim[dim * idx + k] - fluid->fOldDim[dim * idx + k]);
                    }
                }
                // Stream (Bounce Back)
                if (fluid->status1[nextIdx] == status_solid) {
                    fluid->fNewDim[dim * idx + solver->inversed[d]] = combinedf;
                } else {
                    fluid->fNewDim[dim * nextIdx + d] = combinedf;
                }
            } else {
                nextX = (nextX + fluid->nX) % fluid->nX;
                nextY = (nextY + fluid->nY) % fluid->nY;
                nextIdx = getIndex(fluid, nextX, nextY);
                fluid->streamedDim[dim * nextIdx + d] = false;
            }
        }
    }
}

__global__ void kernelUpdateQuantityFine(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        // Initialize
        fluid->density1[idx] = 0;
        if (isBoundary(fluid, x, y) == true) {
            fluid->vel2[2 * idx    ] = 0.5 * fluid->force2[2 * idx];
            fluid->vel2[2 * idx + 1] = 0.5 * fluid->force2[2 * idx + 1];
        } else {
            fluid->vel2[2 * idx    ] = 0.5 * fluid->force2[2 * idx];
            fluid->vel2[2 * idx + 1] = 0.5 * fluid->force2[2 * idx + 1];
        }
        // Density and Velocity
        for (int d = 0; d < dim; ++d) {
            fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        fluid->vel2[2 * idx    ] = fluid->vel2[2 * idx    ] / fluid->density1[idx];
        fluid->vel2[2 * idx + 1] = fluid->vel2[2 * idx + 1] / fluid->density1[idx];
    }
}

__global__ void kernelUpdateBubbleQuantityFine(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        // Initialize
        fluid->density1[idx] = 0;
        fluid->vel2[2 * idx    ] = 0.5 * fluid->force2[2 * idx    ];
        fluid->vel2[2 * idx + 1] = 0.5 * fluid->force2[2 * idx + 1];
        fluid->kxy[idx] = 0;
        fluid->kxx[idx] = 0;
        fluid->kyy[idx] = 0;
        fluid->pix[idx] = 0;
        fluid->piy[idx] = 0;
        // Density and Velocity (without adding 1/2 force)
        for (int d = 0; d < dim; ++d) {
            fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        fluid->vel2[2 * idx    ] = fluid->vel2[2 * idx    ] / fluid->density1[idx];
        fluid->vel2[2 * idx + 1] = fluid->vel2[2 * idx + 1] / fluid->density1[idx];
        // kxy, kxx and kyy
        for (int d = 0; d < dim; ++d) {
            fluid->kxy[idx] += (solver->ciX[d] - fluid->vel2[2 * idx]) * (solver->ciY[d] - fluid->vel2[2 * idx + 1]) * fluid->fNewDim[dim * idx + d];
            fluid->kxx[idx] += pow(solver->ciX[d] - fluid->vel2[2 * idx], 2) * fluid->fNewDim[dim * idx + d];
            fluid->kyy[idx] += pow(solver->ciY[d] - fluid->vel2[2 * idx + 1], 2) * fluid->fNewDim[dim * idx + d];
        }
        for (int d = 0; d < dim; ++d) {
            fluid->pix[idx] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->piy[idx] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        // if (abs(fluid->density1[idx] - 1.0f) >= 0.1) printf(" %f ", fluid->density1[idx]);
        
        
        // // Update velocity (adding 1/2 force)
        // fluid->vel2[2 * idx    ] += 0.5 * (fluid->force2[2 * idx    ]) / fluid->density1[idx];
        // fluid->vel2[2 * idx + 1] += 0.5 * (fluid->force2[2 * idx + 1]) / fluid->density1[idx];

        // // Initialize
        // fluid->density1[idx] = 0;
        // fluid->vel2[2 * idx    ] = 0;
        // fluid->vel2[2 * idx + 1] = 0;
        // fluid->kxy[idx] = 0;
        // fluid->kxx[idx] = 0;
        // fluid->kyy[idx] = 0;
        // fluid->pix[idx] = 0;
        // fluid->piy[idx] = 0;
        // // Density and Velocity (without adding 1/2 force)
        // for (int d = 0; d < dim; ++d) {
        //     fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
        //     fluid->vel2[2 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
        //     fluid->vel2[2 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        // }
        // fluid->vel2[2 * idx    ] = fluid->vel2[2 * idx    ] / fluid->density1[idx];
        // fluid->vel2[2 * idx + 1] = fluid->vel2[2 * idx + 1] / fluid->density1[idx];
        // // kxy, kxx and kyy
        // for (int d = 0; d < dim; ++d) {
        //     fluid->kxy[idx] += (solver->ciX[d] - fluid->vel2[2 * idx]) * (solver->ciY[d] - fluid->vel2[2 * idx + 1]) * fluid->fNewDim[dim * idx + d];
        //     fluid->kxx[idx] += pow(solver->ciX[d] - fluid->vel2[2 * idx], 2) * fluid->fNewDim[dim * idx + d];
        //     fluid->kyy[idx] += pow(solver->ciY[d] - fluid->vel2[2 * idx + 1], 2) * fluid->fNewDim[dim * idx + d];
        // }
        // for (int d = 0; d < dim; ++d) {
        //     fluid->pix[idx] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
        //     fluid->piy[idx] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        // }
        // // Update velocity (adding 1/2 force)
        // fluid->vel2[2 * idx    ] += 0.5 * (fluid->force2[2 * idx    ]) / fluid->density1[idx];
        // fluid->vel2[2 * idx + 1] += 0.5 * (fluid->force2[2 * idx + 1]) / fluid->density1[idx];
    }
}

__global__ void kernelUpdateBubble2018QuantityFine(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        // Initialize
        fluid->density1[idx] = 0;
        fluid->vel2[2 * idx    ] = 0;
        fluid->vel2[2 * idx + 1] = 0;
        fluid->kxy[idx] = 0;
        fluid->kxx[idx] = 0;
        fluid->kyy[idx] = 0;
        fluid->pix[idx] = 0;
        fluid->piy[idx] = 0;
        fluid->c20[idx] = 0;
        fluid->c02[idx] = 0;
        fluid->c11[idx] = 0;
        // Density and Velocity (without adding 1/2 force)
        for (int d = 0; d < dim; ++d) {
            fluid->density1[idx] += fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx    ] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->vel2[2 * idx + 1] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        fluid->vel2[2 * idx    ] = fluid->vel2[2 * idx    ] / fluid->density1[idx];
        fluid->vel2[2 * idx + 1] = fluid->vel2[2 * idx + 1] / fluid->density1[idx];
        // kxy, kxx and kyy
        for (int d = 0; d < dim; ++d) {
            fluid->kxy[idx] += (solver->ciX[d] - fluid->vel2[2 * idx]) * (solver->ciY[d] - fluid->vel2[2 * idx + 1]) * fluid->fNewDim[dim * idx + d];
            fluid->kxx[idx] += pow(solver->ciX[d] - fluid->vel2[2 * idx], 2) * fluid->fNewDim[dim * idx + d];
            fluid->kyy[idx] += pow(solver->ciY[d] - fluid->vel2[2 * idx + 1], 2) * fluid->fNewDim[dim * idx + d];
        }
        for (int d = 0; d < dim; ++d) {
            fluid->pix[idx] += solver->ciX[d] * fluid->fNewDim[dim * idx + d];
            fluid->piy[idx] += solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        // c20, c02 and c11
        for (int d = 0; d < dim; ++d) {
            fluid->c20[idx] += pow(solver->ciX[d], 2) * fluid->fNewDim[dim * idx + d];
            fluid->c02[idx] += pow(solver->ciY[d], 2) * fluid->fNewDim[dim * idx + d];
            fluid->c11[idx] += solver->ciX[d] * solver->ciY[d] * fluid->fNewDim[dim * idx + d];
        }
        fluid->c20[idx] = fluid->c20[idx] / fluid->density1[idx] - pow(fluid->vel2[2 * idx], 2);
        fluid->c02[idx] = fluid->c02[idx] / fluid->density1[idx] - pow(fluid->vel2[2 * idx + 1], 2);
        fluid->c11[idx] = fluid->c11[idx] / fluid->density1[idx] - fluid->vel2[2 * idx] * fluid->vel2[2 * idx + 1];

        fluid->vel2[2 * idx    ] += 0.5 * fluid->force2[2 * idx    ] / fluid->density1[idx];
        fluid->vel2[2 * idx + 1] += 0.5 * fluid->force2[2 * idx + 1] / fluid->density1[idx];
    }
}

__global__ void kernelCalInertiaForceFine(Solver * solver, Fluid * fluid, int t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    if (inRange(fluid, x, y) && fluid->status1[idx] == status_fluid) {
        float accOfReferFrameX = - fluid->accX;
        float accOfReferFrameY = - fluid->accY;

        float eulerAccX = - (0 - fluid->angAccZ * (y - fluid->nY/2.0));
        float eulerAccY = - (fluid->angAccZ * (x - fluid->nX/2.0) - 0);

        float centrifugalAccTempX = (0 - fluid->angVelZ * (y - fluid->nY/2.0));
        float centrifugalAccTempY = (fluid->angVelZ * (x - fluid->nX/2.0) - 0);
    
        float centrifugalAccX = - (0 - fluid->angVelZ * centrifugalAccTempY);
        float centrifugalAccY = - (fluid->angVelZ * centrifugalAccTempX - 0);

        float coriolisAccX = - 2 * (0 - fluid->angVelZ * fluid->vel2[2 * idx + 1]);
        float coriolisAccY = - 2 * (fluid->angVelZ * fluid->vel2[2 * idx] - 0);

        fluid->force2[2 * idx    ] += accOfReferFrameX + eulerAccX + centrifugalAccX + coriolisAccX;
        fluid->force2[2 * idx + 1] += accOfReferFrameY + eulerAccY + centrifugalAccY + coriolisAccY;
    }
}

__global__ void kernelInterpolateFine(Solver * solver, Fluid * fluid, Solver * sC, Fluid * fC, int t, float interpolateRatio)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = getIndex(fluid, x, y);

    float csSquareInv = 3;

    // Only the boundary nodes will be streamed
    if (isBoundary(fluid, x, y)) {
        float posInFineCoorX = (x - fluid->nX / 2.0f) / fluid->scale;
        float posInFineCoorY = (y - fluid->nY / 2.0f) / fluid->scale;

        float posInCoarseCoorX = fluid->rotMatrix[0] * posInFineCoorX + fluid->rotMatrix[1] * posInFineCoorY + fluid->posX;
        float posInCoarseCoorY = fluid->rotMatrix[3] * posInFineCoorX + fluid->rotMatrix[4] * posInFineCoorY + fluid->posY;

        // Idx of Bottom West South node
        int BWSidxX = (int) posInCoarseCoorX;
        int BWSidxY = (int) posInCoarseCoorY;

        // fOldDim is cleaned for storage reuse
        for (int d = 0; d < dim; d++) {
            fluid->fOldDim[dim * idx + d] = 0;
            if (fluid->streamedDim[dim * idx + d] == false) {
                fluid->fNewDim[dim * idx + d] = 0;
            }
        }

        for (int cY = BWSidxY; cY <= BWSidxY + 1; cY++) {
            for (int cX = BWSidxX; cX <= BWSidxX + 1; cX++) {
                assert(inRange(fC, cX, cY));
                int neighborIdx = getIndex(fC, cX, cY);

                float distX = abs(posInCoarseCoorX - (float) cX);
                float distY = abs(posInCoarseCoorY - (float) cY);

                float weight = delta(distX) * delta(distY);

                // Angular speed in coarse grid = angular speed in fine grid * scale
                float angVelZCoarse = fluid->angVelZ * fluid->scale;

                // Grid Vel = vel + w.cross(radius), the grid velocity is measured in coarse coordinate
                float gridVelX = fluid->velX + calVecCrossX(0, 0, angVelZCoarse, cX - fluid->posX, cY - fluid->posY, 0);
                float gridVelY = fluid->velY + calVecCrossY(0, 0, angVelZCoarse, cX - fluid->posX, cY - fluid->posY, 0);

                float uxAbs = interpolateRatio * fC->velOld2[2 * neighborIdx    ] + (1 - interpolateRatio) * fC->vel2[2 * neighborIdx    ];
                float uyAbs = interpolateRatio * fC->velOld2[2 * neighborIdx + 1] + (1 - interpolateRatio) * fC->vel2[2 * neighborIdx + 1];

                float uxRel = uxAbs - gridVelX;
                float uyRel = uyAbs - gridVelY;

                float dens = interpolateRatio * fC->densityOld1[neighborIdx] + (1 - interpolateRatio) * fC->density1[neighborIdx];

                float term1, term2;
                for (int d = 0; d < dim; d++) {
                    // Calculate fa(eq)
                    term1 = solver->ciX[d] * uxAbs + solver->ciY[d] * uyAbs;
                    term2 = uxAbs * uxAbs + uyAbs * uyAbs;
                    float faEqu = solver->w[d] * dens * (1 + csSquareInv * term1 + 0.5 * pow(csSquareInv, 2) * pow(term1, 2) - 0.5 * csSquareInv * term2);
                
                    // Calculate fr(eq)
                    term1 = solver->ciX[d] * uxRel + solver->ciY[d] * uyRel;
                    term2 = uxRel * uxRel + uyRel * uyRel;
                    float frEqu = solver->w[d] * dens * (1 + csSquareInv * term1 + 0.5 * pow(csSquareInv, 2) * pow(term1, 2) - 0.5 * csSquareInv * term2);
                
                    // fr = fa - (fa(eq) - fr(eq))
                    float fInterpolate = interpolateRatio * fC->fLastDim[dim * neighborIdx + d] + (1 - interpolateRatio) * fC->fNewDim[dim * neighborIdx + d];
                    fluid->fOldDim[dim * idx + d] += (fInterpolate - (faEqu - frEqu)) * weight;
                }
            }
        }

        for (int d = 0; d < dim; d++) {
            if (! fluid->streamedDim[dim * idx + d]) {
                for (int k = 0; k < dim; k++) {
                    fluid->fNewDim[dim * idx + d] += solver->MinvMrot[dim * d + k] * fluid->fOldDim[dim * idx + k];
                }
            }
        }
    }
}

// ---------- Main function ---------- 
void colliAndAdvectFine(Solver ** gpuSolverList, Fluid ** gpuFluidList, Solver ** solverList, Fluid ** fluidList, int t, int fluIdx, float interpolateRatio, int interpolateType)
{
    dim3 dimGrid(ceil((float) fluidList[fluIdx]->nX / (float) tileSize), 
                 ceil((float) fluidList[fluIdx]->nY / (float) tileSize), 
                 1);
    dim3 dimBlock(tileSize, tileSize, 1);

    kernelUpdateForceFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx], t);
    cudaCheck(cudaDeviceSynchronize());

    kernelCalInertiaForceFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx], t);
    cudaCheck(cudaDeviceSynchronize());

    kernelFcopyFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx]);
    cudaCheck(cudaDeviceSynchronize());

    kernelForceFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx], t);
    cudaCheck(cudaDeviceSynchronize());

    kernelEquFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx]);
    cudaCheck(cudaDeviceSynchronize());

    kernelColliStreamFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx]);
    cudaCheck(cudaDeviceSynchronize());
    
    if (interpolateType == 0) {
        // Bilinear interpolation
        kernelInterpolateFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx], gpuSolverList[0], gpuFluidList[0], t, interpolateRatio);
        cudaCheck(cudaDeviceSynchronize());
        kernelUpdateQuantityFine<<<dimGrid, dimBlock>>>(gpuSolverList[fluIdx], gpuFluidList[fluIdx], t);
        cudaCheck(cudaDeviceSynchronize());
    } 
    else {
        printf("Failure: No interpolation type %d (FineGrid)\n", interpolateType);
    }
}
