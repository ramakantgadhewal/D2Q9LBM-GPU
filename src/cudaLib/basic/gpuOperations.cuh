#ifndef __GPU_OPERATIONS_CUH__
#define __GPU_OPERATIONS_CUH__

#include <solver.hpp>
#include <fluid.hpp>

void gpuSetDevice(int gpuId);
Solver ** gpuSolverInit(int numMovingFluids, Solver ** solverList, Solver ** ptrContainerList);
Fluid ** gpuFluidInit(int numMovingFluids, Fluid ** fluidList, Fluid ** ptrContainerList);

void solverMrotCpyCpu2Gpu(Solver * solver, Solver * ptrContainer);

void fluidProperityCpyCpu2Gpu(Fluid * fluid, Fluid * ptrContainer, Fluid * gpuFluid);
void fluidVel2CpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer);
void fluidFDimCpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer);
void fluidStatus1CpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer);
void fluidStreamedDimCpyGpu2Cpu(Fluid * fluid, Fluid * ptrContainer); 

void gpuCopySolver(Solver * solver, Solver * ptrContainer);
void gpuCopyFluidVel2(Fluid * fluid, Fluid * ptrContainer);
void gpuCopyFluidFOldDim(Fluid * fluid, Fluid * ptrContainer);
void gpuCopyFluidGridForceDim(Fluid * fluid, Fluid * ptrContainer);
void gpuCopyFluidEquDim(Fluid * fluid, Fluid * ptrContainer);

void gpuSolverFree(int numMovingFluids, Solver ** solverList, Solver ** ptrContainerList, Solver ** gpuSolverList);
void gpuFluidFree(int numMovingFluids, Fluid ** fluidList, Fluid ** ptrContainerList, Fluid ** gpuFluidList);

#endif