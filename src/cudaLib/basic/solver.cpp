#include <solver.hpp>
#include <cudaHelper.cuh>

Solver::Solver(float vis)
{
    float relaxTime = 3. * vis + 0.5;
    relaxFreq = 1. / relaxTime;

    wxx = relaxFreq;
    wxy = relaxFreq;
    w1 = relaxFreq;
}

Solver::~Solver()
{
    delete w;
    delete ciX;
    delete ciY;
    delete inversed;
    delete MinvSM;
    delete MinvMrot;
    delete MinvMrotInv;
}

void Solver::init()
{
    w = new float [dim];
    ciX = new int [dim];
    ciY = new int [dim];
    inversed = new int [dim];
    MinvSM = new float [dim * dim];
    MinvMrot = new float [dim * dim];
    MinvMrotInv = new float [dim * dim];

    w[0] = 4./9.;
    w[1] = 1./9.;  w[2] = 1./9.;  w[3] = 1./9.;  w[4] = 1./9.;
    w[5] = 1./36.; w[6] = 1./36.; w[7] = 1./36.; w[8] = 1./36.;

    ciX[0] =  0; ciX[1] =  1; ciX[2] =  0; ciX[3] = -1; ciX[4] =  0; 
    ciX[5] =  1; ciX[6] = -1; ciX[7] = -1; ciX[8] =  1;

    ciY[0] =  0; ciY[1] =  0; ciY[2] =  1; ciY[3] =  0; ciY[4] = -1;
    ciY[5] =  1; ciY[6] =  1; ciY[7] = -1; ciY[8] = -1;

    int inversed[dim] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    inversed[0] = 0; inversed[1] = 3; inversed[2] = 4; inversed[3] = 1; inversed[4]  = 2; 
    inversed[5] = 7; inversed[6] = 8; inversed[7] = 5; inversed[8] = 6;
}




