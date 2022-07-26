#include <fluid.hpp>
#include <cudaHelper.cuh>
#include <cmath>

Fluid::Fluid(int scaleIn, int nXIn, int nYIn, float accXIn, float accYIn, float posXIn, float posYIn, 
          float angAccZIn, float angPosZIn, float invalidSizeXIn, float invalidSizeYIn)
{
    scale = scaleIn;

    nX = nXIn;
    nY = nYIn;
    nAll = nX * nY;

    accX = accXIn; 
    accY = accYIn; 
    velX = 0; 
    velY = 0; 
    posX = posXIn; 
    posY = posYIn; 

    angAccZ = angAccZIn;
    angVelZ = 0;
    angPosZ = angPosZIn;

    invalidSizeX = invalidSizeXIn;
    invalidSizeY = invalidSizeYIn;
}

Fluid::~Fluid()
{
    delete vel2;
    delete force2;
    delete density1;
    delete status1;
    delete equDim;
    delete fNewDim;
    delete fOldDim;
    delete gridForceDim;
    delete streamedDim;
    
    delete rotMatrix;
    delete rotMatrixInv;
}

void Fluid::init(Solver * solver, Shape * shape)
{
    // Quantities for LBM
    vel2         = new float  [2 * nAll];
    force2       = new float  [2 * nAll];
    density1     = new float  [nAll];
    status1      = new Status [nAll];
    fNewDim      = new float  [dim * nAll];
    fOldDim      = new float  [dim * nAll];
    gridForceDim = new float  [dim * nAll];
    equDim       = new float  [dim * nAll];
    streamedDim  = new bool   [dim * nAll];
    // Three moments for bubble function
    kxy = new float [nAll];
    kxx = new float [nAll];
    kyy = new float [nAll];
    pix = new float [nAll];
    piy = new float [nAll];
    c20 = new float [nAll];
    c02 = new float [nAll];
    c11 = new float [nAll];
    // For overset Grid
    rotMatrix = new float [9];
    rotMatrixInv = new float [9];
    // Init quantities
    for (int idx = 0; idx < nX * nY; idx++) {
        // Quantities for LBM
        for (int i = 0; i < 2; i++) {
            vel2  [2 * idx + i] = 0;
            force2[2 * idx + i] = 0;
        }
        density1[idx] = 1;
        status1 [idx] = status_fluid;
        for (int d = 0; d < dim; ++d) {
            float currEqu = solver->w[d] * density1[idx] * (1 + 0 + 0 - 0);
            equDim      [dim * idx + d] = currEqu;
            fNewDim     [dim * idx + d] = currEqu;
            fOldDim     [dim * idx + d] = currEqu;
            gridForceDim[dim * idx + d] = 0;
        }
        // Moments for bubble function
        kxy[idx] = 0;
        kxx[idx] = 0;
        kyy[idx] = 0;
        pix[idx] = 0;
        piy[idx] = 0;
        c20[idx] = 0;
        c02[idx] = 0;
        c11[idx] = 0;
        for (int d = 0; d < dim; ++d) {
            kxy[idx] += solver->ciX[d] * solver->ciY[d] * equDim[d];
            kxx[idx] += pow(solver->ciX[d], 2) * equDim[d];
            kyy[idx] += pow(solver->ciY[d], 2) * equDim[d];
            c20[idx] += pow(solver->ciX[d], 2) * equDim[d];
            c02[idx] += pow(solver->ciY[d], 2) * equDim[d];
            c11[idx] += solver->ciX[d] * solver->ciY[d] * equDim[d];
        }
    }
    // Init shape
    for (int i = 0; i < nAll; i++) {
        if (shape->shapeList[i] == true) {
            status1[i] = status_solid;
        }
    }
}
