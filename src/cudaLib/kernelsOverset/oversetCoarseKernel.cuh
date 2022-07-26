#ifndef __OVERSET_COARSE_KERNEL_CUH__
#define __OVERSET_COARSE_KERNEL_CUH__

#include <solver.hpp>
#include <fluid.hpp>
#include <userDefine.hpp>

void colliAndAdvectCoarse(Solver ** gpuSolverList, Fluid ** gpuFluidList, Solver ** solverList, Fluid ** fluidList, int t, int numFineFluids, int interpolateType);

#endif
