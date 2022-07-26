#ifndef __VTKWRITER_HPP__
#define __VTKWRITER_HPP__

#include <string>

#include <userDefine.hpp>
#include <solver.hpp>
#include <fluid.hpp>

using namespace std;

void findMaxVel(Solver * solver, Fluid * fluid, string project, int time);
void writeFluidDataVtk(Solver * solver, Fluid * fluid, string project, int time);
void writeFluidStatusVtk(Solver * solver, Fluid * fluid, string project, int time);
void writeFluidDataStaticVtk(Solver * solver, Fluid * fluid, string project, int time);
void writeFluidDataIntegratedVtk(Solver ** solverList, Fluid ** fluidList, string project, int time);
void writeFluidDataIntegratedInFineResolutionVtk(Solver ** solverList, Fluid ** fluidList, string project, int time);
void writeFNewDataIntegratedVtk(Solver ** solverList, Fluid ** fluidList, string project, int time);
void writeParticleTraceData(Solver ** solverList, Fluid ** fluidList, string project, int time);
void writeParticleTraceDataInFineResolution(Solver ** solverList, Fluid ** fluidList, string project, int time);

#endif