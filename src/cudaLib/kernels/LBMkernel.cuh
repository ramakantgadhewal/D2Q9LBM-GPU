#ifndef __LBM_KERNEL_CUH__
#define __LBM_KERNEL_CUH__

#include <solver.hpp>
#include <fluid.hpp>
#include <userDefine.hpp>

void colliAndAdvect(Solver * gpuSolver, Fluid * gpuFluid, int t);

#endif