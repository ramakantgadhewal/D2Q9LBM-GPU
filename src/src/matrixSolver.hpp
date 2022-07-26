#ifndef __MATRIX_SOLVER_HPP__
#define __MATRIX_SOLVER_HPP__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <userDefine.hpp>

using namespace std;
using namespace Eigen;

class MatrixSolver
{
public:
    int * ciX;
    int * ciY;

    Matrix<float, dim, dim> S;
    Matrix<float, dim, dim> M;
    Matrix<float, dim, dim> Minv;
    Matrix<float, dim, dim> MinvS;
    Matrix<float, dim, dim> MinvSM;

    Matrix<float, dim, dim> Mrot;
    Matrix<float, dim, dim> MrotInv;
    Matrix<float, dim, dim> MinvMrot;
    Matrix<float, dim, dim> MinvMrotInv;

    Matrix3f rotMatrix;
    Matrix3f rotMatrixInv;

    MatrixSolver(float vis);
    ~MatrixSolver() {};

    void updateRotateMatrix(float angleInZ);
    void updateMrotate();
};

#endif