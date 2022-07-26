#ifndef __FLUID_HPP__
#define __FLUID_HPP__

#include <userDefine.hpp>
#include <solver.hpp>
#include <shape.hpp>

enum Status 
{
    status_fluid, 
    status_solid, 
    status_invalid
};

class Fluid
{
public:

    // Base properties
    int nX, nY, nAll;
    // Quantities for LBM
    float * vel2;
    float * force2;
    float * density1;
    Status * status1;
    float * fNewDim;
    float * fOldDim;
    float * gridForceDim;
    float * equDim;
    bool * streamedDim;
    // Three moments for bubble function
    float * kxy;
    float * kxx;
    float * kyy;
    float * pix;
    float * piy;
    float * c20;
    float * c02;
    float * c11;
    // Redundant base properties for coarse grid time interpolation
    float * densityOld1;
    float * velOld2;
    float * fLastDim;
    // For overset Grid
    int scale;
    float accX, accY;
    float velX, velY;
    float posX, posY;
    float angAccZ;
    float angVelZ;
    float angPosZ;
    float invalidSizeX, invalidSizeY;
    float * rotMatrix;
    float * rotMatrixInv;

    Fluid(int scaleIn, int nXIn, int nYIn, float accXIn, float accYIn, float posXIn, float posYIn, 
          float angAccZIn, float angPosZIn, float invalidSizeXIn, float invalidSizeYIn);
    ~Fluid();
    void init(Solver * solver, Shape * shape);
    void initStreamed(Solver * solver);
};

#endif