#include <gpuOperations.cuh>
#include <cudaHelper.cuh>

void gpuSetDevice(int gpuId)
{
    int gpuCount;
    cudaCheck(cudaGetDeviceCount(& gpuCount));
    if (gpuCount == 0) {
        fprintf(stderr, "There is no CUDA device.\n");
    }
    if (gpuId >= gpuCount) {
        fprintf(stderr, "GPU id is %d. But there are only %d GPUs.\n", gpuId, gpuCount);
    }

    cudaCheck(cudaSetDevice(gpuId));

    printf("Total number of GPUs: %d, Using GPU id: %d\n", gpuCount, gpuId);
}

Solver ** gpuSolverInit(int numMovingFluids, Solver ** solverList, Solver ** ptrContainerList)
{
    Solver ** gpuSolverList = new Solver * [numMovingFluids + 1];

    for (int i = 0; i < numMovingFluids + 1; i++) {
        cudaCheck(cudaMalloc(& ptrContainerList[i]->w,           dim * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->ciX,         dim * sizeof(int)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->ciY,         dim * sizeof(int)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->inversed,    dim * sizeof(int)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->MinvSM,      dim * dim * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->MinvMrot,    dim * dim * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->MinvMrotInv, dim * dim * sizeof(float)));

        cudaCheck(cudaMemcpy(ptrContainerList[i]->w,           solverList[i]->w,           dim * sizeof(float),       cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->ciX,         solverList[i]->ciX,         dim * sizeof(int),         cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->ciY,         solverList[i]->ciY,         dim * sizeof(int),         cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->inversed,    solverList[i]->inversed,    dim * sizeof(int),         cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->MinvSM,      solverList[i]->MinvSM,      dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->MinvMrot,    solverList[i]->MinvMrot,    dim * dim * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->MinvMrotInv, solverList[i]->MinvMrotInv, dim * dim * sizeof(float), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc(& gpuSolverList[i], sizeof(Solver)));
        cudaCheck(cudaMemcpy(gpuSolverList[i], ptrContainerList[i], sizeof(Solver), cudaMemcpyHostToDevice));
    }

    return gpuSolverList;
}

Fluid ** gpuFluidInit(int numMovingFluids, Fluid ** fluidList, Fluid ** ptrContainerList)
{
    Fluid ** gpuFluidList = new Fluid * [numMovingFluids + 1];

    for (int i = 0; i < numMovingFluids + 1; i++) {
        int currNAll = fluidList[i]->nAll;
        // Quantities for LBM
        cudaCheck(cudaMalloc(& ptrContainerList[i]->density1,           currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->status1,            currNAll * sizeof(Status)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->vel2,           2 * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->force2,         2 * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->fNewDim,      dim * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->fOldDim,      dim * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->gridForceDim, dim * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->equDim,       dim * currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->streamedDim,  dim * currNAll * sizeof(bool)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->rotMatrix,                9 * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->rotMatrixInv,             9 * sizeof(float)));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->density1,     fluidList[i]->density1,           currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->status1,      fluidList[i]->status1 ,           currNAll * sizeof(Status), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->vel2,         fluidList[i]->vel2 ,          2 * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->force2,       fluidList[i]->force2 ,        2 * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->fNewDim,      fluidList[i]->fNewDim ,     dim * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->fOldDim,      fluidList[i]->fOldDim ,     dim * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->gridForceDim, fluidList[i]->gridForceDim, dim * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->equDim,       fluidList[i]->equDim ,      dim * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->streamedDim,  fluidList[i]->streamedDim,  dim * currNAll * sizeof(bool),  cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->rotMatrix,    fluidList[i]->rotMatrix,                 9 * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->rotMatrixInv, fluidList[i]->rotMatrixInv,              9 * sizeof(float), cudaMemcpyHostToDevice));
        // Three moments for bubble function
        cudaCheck(cudaMalloc(& ptrContainerList[i]->kxy, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->kxx, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->kyy, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->pix, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->piy, currNAll * sizeof(float)));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->kxy, fluidList[i]->kxy, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->kxx, fluidList[i]->kxx, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->kyy, fluidList[i]->kyy, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        // Three moments for 2018 bubble function
        cudaCheck(cudaMalloc(& ptrContainerList[i]->c20, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->c02, currNAll * sizeof(float)));
        cudaCheck(cudaMalloc(& ptrContainerList[i]->c11, currNAll * sizeof(float)));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->c20, fluidList[i]->c20, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->c02, fluidList[i]->c02, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(ptrContainerList[i]->c11, fluidList[i]->c11, currNAll * sizeof(float), cudaMemcpyHostToDevice));
        // For overset Grid
        if (i == 0) {
            cudaCheck(cudaMalloc(& ptrContainerList[i]->densityOld1,    currNAll * sizeof(float)));
            cudaCheck(cudaMalloc(& ptrContainerList[i]->velOld2,    2 * currNAll * sizeof(float)));
            cudaCheck(cudaMalloc(& ptrContainerList[i]->fLastDim, dim * currNAll * sizeof(float)));
            cudaCheck(cudaMemcpy(ptrContainerList[i]->densityOld1, fluidList[i]->density1,      currNAll * sizeof(float), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(ptrContainerList[i]->velOld2,     fluidList[i]->vel2,      2 * currNAll * sizeof(float), cudaMemcpyHostToDevice));
            cudaCheck(cudaMemcpy(ptrContainerList[i]->fLastDim,    fluidList[i]->fNewDim, dim * currNAll * sizeof(float), cudaMemcpyHostToDevice));
        }
        // Combined together
        cudaCheck(cudaMalloc(& gpuFluidList[i], sizeof(Fluid)));
        cudaCheck(cudaMemcpy(gpuFluidList[i], ptrContainerList[i], sizeof(Fluid), cudaMemcpyHostToDevice));
    }

    return gpuFluidList;
}

void gpuSolverFree(int numMovingFluids, Solver ** solverList, Solver ** ptrContainerList, Solver ** gpuSolverList)
{
    for (int i = 0; i < numMovingFluids + 1; i++) {
        cudaCheck(cudaFree(ptrContainerList[i]->w));
        cudaCheck(cudaFree(ptrContainerList[i]->ciX));
        cudaCheck(cudaFree(ptrContainerList[i]->ciY));
        cudaCheck(cudaFree(ptrContainerList[i]->inversed));
        cudaCheck(cudaFree(ptrContainerList[i]->MinvSM));
        cudaCheck(cudaFree(ptrContainerList[i]->MinvMrot));
        cudaCheck(cudaFree(ptrContainerList[i]->MinvMrotInv));

        cudaCheck(cudaFree(gpuSolverList[i]));
    }
}

void gpuFluidFree(int numMovingFluids, Fluid ** fluidList, Fluid ** ptrContainerList, Fluid ** gpuFluidList)
{
    for (int i = 0; i < numMovingFluids + 1; i++) {
        cudaCheck(cudaFree(ptrContainerList[i]->density1));
        cudaCheck(cudaFree(ptrContainerList[i]->status1));
        cudaCheck(cudaFree(ptrContainerList[i]->vel2));
        cudaCheck(cudaFree(ptrContainerList[i]->force2));
        cudaCheck(cudaFree(ptrContainerList[i]->fNewDim));
        cudaCheck(cudaFree(ptrContainerList[i]->fOldDim));
        cudaCheck(cudaFree(ptrContainerList[i]->gridForceDim));
        cudaCheck(cudaFree(ptrContainerList[i]->equDim));
        cudaCheck(cudaFree(ptrContainerList[i]->streamedDim));
        cudaCheck(cudaFree(ptrContainerList[i]->rotMatrix));
        cudaCheck(cudaFree(ptrContainerList[i]->rotMatrixInv));

        if (i == 0) {
            cudaCheck(cudaFree(ptrContainerList[i]->densityOld1));
            cudaCheck(cudaFree(ptrContainerList[i]->velOld2));
            cudaCheck(cudaFree(ptrContainerList[i]->fLastDim));
        }

        cudaCheck(cudaFree(gpuFluidList[i]));
    }
}

void solverMrotCpyCpu2Gpu(Solver * solver, Solver * ptrContainer)
{
    cudaCheck(cudaMemcpy(ptrContainer->MinvMrot,    solver->MinvMrot,    dim * dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(ptrContainer->MinvMrotInv, solver->MinvMrotInv, dim * dim * sizeof(float), cudaMemcpyHostToDevice));
}

void fluidProperityCpyCpu2Gpu(Fluid * fluid, Fluid * ptrContainer, Fluid * gpuFluid)
{
    cudaCheck(cudaMemcpy(ptrContainer->rotMatrix,    fluid->rotMatrix,    9 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(ptrContainer->rotMatrixInv, fluid->rotMatrixInv, 9 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(gpuFluid, ptrContainer, sizeof(Fluid), cudaMemcpyHostToDevice));
}

void fluidVel2CpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->vel2, ptrContainer->vel2, 2 * fluid->nAll * sizeof(float), cudaMemcpyDeviceToHost));
}

void fluidFDimCpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->fNewDim, ptrContainer->fNewDim, dim * fluid->nAll * sizeof(float), cudaMemcpyDeviceToHost));
}

void fluidStatus1CpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->status1, ptrContainer->status1, fluid->nAll * sizeof(Status), cudaMemcpyDeviceToHost));
}

void fluidStreamedDimCpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->streamedDim, ptrContainer->streamedDim, dim * fluid->nAll * sizeof(bool), cudaMemcpyDeviceToHost));
}

// ---------- For data output ----------
void gpuCopyFluidFOldDim(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->fOldDim, ptrContainer->fOldDim, dim * fluid->nAll * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpuCopyFluidGridForceDim(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->gridForceDim, ptrContainer->gridForceDim, dim * fluid->nAll * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpuCopyFluidEquDim(Fluid * fluid, Fluid * ptrContainer)
{
    cudaCheck(cudaMemcpy(fluid->equDim, ptrContainer->equDim, dim * fluid->nAll * sizeof(float), cudaMemcpyDeviceToHost));
}

// ---------- For overset grid ----------
void solverCpy2Gpu(Solver * solver, Solver * ptrContainer)
{
    cudaCheck(cudaMemcpy(ptrContainer->ciX, solver->ciX, dim * sizeof(int), cudaMemcpyHostToDevice));
}
