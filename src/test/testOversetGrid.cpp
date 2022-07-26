#include <cmath>
#include <iostream>
#include <cstdlib> 

#include <userDefine.hpp>
#include <solver.hpp>
#include <fluid.hpp>
#include <shape.hpp>
#include <gpuOperations.cuh>
#include <vtkWriter.hpp>
#include <matrixSolver.hpp>
#include <rawDataWriter.hpp>

#include <oversetFineKernel.cuh>
#include <oversetCoarseKernel.cuh>

using namespace std;

// Function declarations
int main();

int main()
{
    /////////////////////////////////////////////////////
    // ---------- Project name, for data print ----------
    /////////////////////////////////////////////////////
    string project = "LBM-OversetGrid";
    
    /////////////////////////////////////////////////////
    // ---------- Set fluid property ----------
    /////////////////////////////////////////////////////
    // ID of using GPU
    int gpuId = 0;
    // The invalid region is measured in staticFluid scale
    int staticFluidSizeX = 800;
    int staticFluidSizeY = 800;
    /*float vis = 0.00036;*/
    float vis = 0.0001;

    /////////////////////////////////////////////////////
    // ---------- Init shape ----------
    /////////////////////////////////////////////////////
    cout << "Shape initing. ";
    Shape * blankStatic = new Shape(staticFluidSizeX, staticFluidSizeY);

    // ---------- Init matrixSolver ----------
    cout << "Begin matrixSolver initialization.\n";
    MatrixSolver* matrixSolver = new MatrixSolver(vis);
    cout << "MatrixSolver initialization done.\n";

    /////////////////////////////////////////////////////
    // ---------- Init solver ----------
    /////////////////////////////////////////////////////
    cout << "Solver initing. ";
    Solver ** solverList = new Solver * [1]; 
    solverList[0] = new Solver(vis);
    solverList[0]->init();
    for (int y = 0; y < dim; y++) {
        for (int x = 0; x < dim; x++) {
            int idx = y * dim + x;
            solverList[0]->MinvSM[idx] = matrixSolver->MinvSM(y, x);
        }
    }
    cout << "Solver init done.\n";

    /////////////////////////////////////////////////////
    // ---------- Init fluid ---------- 
    /////////////////////////////////////////////////////
    cout << "Fluid initing. ";
    Fluid ** fluidList = new Fluid * [1];
    fluidList[0] = new Fluid(1, staticFluidSizeX, staticFluidSizeY, 0, 0, 0, 0, 0, 0, 0, 0);
    fluidList[0]->init(solverList[0], blankStatic);
    cout << "Fluid init done.\n";

    /////////////////////////////////////////////////////
    // ---------- Init ptrContainer ----------
    /////////////////////////////////////////////////////
    cout << "PtrContainer initing. ";
    Solver ** solverPtrContainerList = new Solver * [1]; 
    Fluid ** fluidPtrContainerList = new Fluid * [1];
    solverPtrContainerList[0] = new Solver(vis);
    fluidPtrContainerList[0] = new Fluid(1, staticFluidSizeX, staticFluidSizeY, 0, 0, 0, 0, 0, 0, 0, 0);
    cout << "PtrContainer init done.\n";

    /////////////////////////////////////////////////////
    // ---------- Init CUDA ----------
    /////////////////////////////////////////////////////
    cout << "GPU initing. ";
    gpuSetDevice(gpuId);
    Solver ** gpuSolverList = gpuSolverInit(0, solverList, solverPtrContainerList);
    Fluid  ** gpuFluidList  = gpuFluidInit(0, fluidList, fluidPtrContainerList);
    cout << "GPU init done.\n";

    /////////////////////////////////////////////////////
    // ---------- Print information ----------
    /////////////////////////////////////////////////////
    for (int i = 0; i < 1; i++) {
        cout << "Grid size " << i << " : " << fluidList[i]->nX << " " << fluidList[i]->nY << "  ";
        cout << "Grid dimension: " << ceil((float) fluidList[i]->nX / (float) tileSize) << " " << ceil((float) fluidList[i]->nY / (float) tileSize) << "  ";
        cout << "\n";
    }
    cout << "CUDA block dimension: " << tileSize << "\n";

    /////////////////////////////////////////////////////
    // ---------- Simulation ----------
    /////////////////////////////////////////////////////
    int simuTime = 60000;
    /*for (int t = 0; t <= 100; t++) {
        if (t % 10 == 0) {
            colliAndAdvectCoarse(gpuSolverList, gpuFluidList, solverList, fluidList, t, 0, 0);
            fluidVel2CpyGpu2Cpu(fluidList[0], fluidPtrContainerList[0]);
            writeFluidDataVtk(solverList[0], fluidList[0], project, t);
        }
    }*/

    for (int t = 0; t <= simuTime; t++) {
        colliAndAdvectCoarse(gpuSolverList, gpuFluidList, solverList, fluidList, t, 0, 0);

        if (t % 200 == 0) {
            fluidVel2CpyGpu2Cpu(fluidList[0], fluidPtrContainerList[0]);
            fluidStatus1CpyGpu2Cpu(fluidList[0], fluidPtrContainerList[0]); 
            fluidStreamedDimCpyGpu2Cpu(fluidList[0], fluidPtrContainerList[0]); 

            writeFluidDataVtk(solverList[0], fluidList[0], project, t);
            //writeFluidDataStaticVtk(solverList[1], fluidList[1], project, t);
            //writeFluidDataIntegratedVtk(solverList, fluidList, project, t);
            //writeFluidDataIntegratedInFineResolutionVtk(solverList, fluidList, project, t);
            //writeParticleTraceData(solverList, fluidList, project, t);
            //writeParticleTraceDataInFineResolution(solverList, fluidList, project, t);

            cout << "accomplished " << t << " of " << simuTime << endl;
        }
    }

    /////////////////////////////////////////////////////
    // ---------- Free space ----------
    /////////////////////////////////////////////////////
    gpuSolverFree(0, solverList, solverPtrContainerList, gpuSolverList);
    gpuFluidFree(0, fluidList, fluidPtrContainerList, gpuFluidList);

    for (int i = 0; i < 1; i++) {
        delete solverList[i];
        delete fluidList[i];
    }
    
    return 0;
}
