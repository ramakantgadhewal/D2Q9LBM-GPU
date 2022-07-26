#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <userDefine.hpp>


class Solver
{
//        0  1  2  3  4  5  6  7  8
// --------------------------------
// x:     0 +1  0 -1  0 +1 -1 -1 +1
// y:     0  0 +1  0 -1 +1 +1 -1 -1
public:
    // Parameters of the fluid
    float relaxFreq;
    float wxx;
    float wxy;
    float w1;
    // Parameters of LBM
    float * w;
    int   * ciX;
    int   * ciY;
    int   * inversed;
    // For collision model
    float * MinvSM;
    float * MinvMrot;
    float * MinvMrotInv;

    Solver(float vis);
    ~Solver();
    void init();
};

#endif