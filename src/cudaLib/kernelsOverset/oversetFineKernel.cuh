#ifndef __OVERSET_FINE_KERNEL_CUH__
#define __OVERSET_FINE_KERNEL_CUH__

#include <solver.hpp>
#include <fluid.hpp>
#include <userDefine.hpp>

void colliAndAdvectFine(Solver ** gpuSolverList, Fluid ** gpuFluidList, Solver ** solverList, Fluid ** fluidList, int t, int fluIdx, float interpolateRatio, int interpolateType);

#endif
