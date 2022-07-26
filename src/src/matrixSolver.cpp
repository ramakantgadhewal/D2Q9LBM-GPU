#include <matrixSolver.hpp>

MatrixSolver::MatrixSolver(float vis)
{
    ciX = new int [dim];
    ciY = new int [dim];

    ciX[0] =  0; ciX[1] =  1; ciX[2] =  0; ciX[3] = -1; ciX[4] =  0; 
    ciX[5] =  1; ciX[6] = -1; ciX[7] = -1; ciX[8] =  1;

    ciY[0] =  0; ciY[1] =  0; ciY[2] =  1; ciY[3] =  0; ciY[4] = -1;
    ciY[5] =  1; ciY[6] =  1; ciY[7] = -1; ciY[8] = -1;

    for (int d = 0; d < dim; d++) {
        float ciXsquare = pow(ciX[d], 2);
        float ciYsquare = pow(ciY[d], 2);
        float squareSum = ciXsquare + ciYsquare;

        M(0, d) = 1;
        M(1, d) = -4 + 3 * squareSum;
        M(2, d) = 4 - 10.5 * squareSum + 4.5 * pow(squareSum, 2);
        M(3, d) = ciX[d];
        M(4, d) = (-5 + 3 * squareSum) * ciX[d];
        M(5, d) = ciY[d];
        M(6, d) = (-5 + 3 * squareSum) * ciY[d];
        M(7, d) = ciXsquare - ciYsquare;
        M(8, d) = ciX[d] * ciY[d];
    }

    float relaxFreq = 1.0 / (3.0 * vis + 0.5);

    for (int i = 0; i < dim; i++) {
        S(i, i) = relaxFreq;
    }

    Minv = M.inverse();
    MinvS = Minv * S;
    MinvSM = Minv * S * M;

    updateRotateMatrix(0);
    updateMrotate();
};

void MatrixSolver::updateRotateMatrix(float angleInZ)
{
    rotMatrix << cos(angleInZ), -sin(angleInZ), 0,
                 sin(angleInZ),  cos(angleInZ), 0,
                 0,           0,          1;

    rotMatrixInv = rotMatrix.inverse();
}

void MatrixSolver::updateMrotate()
{
    // ci rotates to negative direction corresponds to fNew rotate to positive direction, 
    // because ci is the cordinate system.
    // This function must be called after updateRotateMatrix.

    /////////////////////////////
    // ---------- Mrot ----------
    /////////////////////////////
    float rotatedCiX[dim];
    float rotatedCiY[dim];

    for (int d = 0; d < dim; d++) {
        Vector3f axis, rotAxis;
        axis << ciX[d], ciY[d], 0;

        rotAxis = rotMatrixInv * axis;

        rotatedCiX[d] = rotAxis(0);
        rotatedCiY[d] = rotAxis(1);
    }

    for (int d = 0; d < dim; d++) {
        float ciXsquare = pow(rotatedCiX[d], 2);
        float ciYsquare = pow(rotatedCiY[d], 2);
        float squareSum = ciXsquare + ciYsquare;

        Mrot(0, d) = 1;
        Mrot(1, d) = -4 + 3 * squareSum;
        Mrot(2, d) = 4 - 10.5 * squareSum + 4.5 * pow(squareSum, 2);
        Mrot(3, d) = rotatedCiX[d];
        Mrot(4, d) = (-5 + 3 * squareSum) * rotatedCiX[d];
        Mrot(5, d) = rotatedCiY[d];
        Mrot(6, d) = (-5 + 3 * squareSum) * rotatedCiY[d];
        Mrot(7, d) = ciXsquare - ciYsquare;
        Mrot(8, d) = rotatedCiX[d] * rotatedCiY[d];
    }

    ////////////////////////////////
    // ---------- MrotInv ----------
    ////////////////////////////////
    for (int d = 0; d < dim; d++) {
        Vector3f axis, rotAxis;
        axis << ciX[d], ciY[d], 0;

        rotAxis = rotMatrix * axis;

        rotatedCiX[d] = rotAxis(0);
        rotatedCiY[d] = rotAxis(1);
    }

    for (int d = 0; d < dim; d++) {
        float ciXsquare = pow(rotatedCiX[d], 2);
        float ciYsquare = pow(rotatedCiY[d], 2);
        float squareSum = ciXsquare + ciYsquare;

        MrotInv(0, d) = 1;
        MrotInv(1, d) = -4 + 3 * squareSum;
        MrotInv(2, d) = 4 - 10.5 * squareSum + 4.5 * pow(squareSum, 2);
        MrotInv(3, d) = rotatedCiX[d];
        MrotInv(4, d) = (-5 + 3 * squareSum) * rotatedCiX[d];
        MrotInv(5, d) = rotatedCiY[d];
        MrotInv(6, d) = (-5 + 3 * squareSum) * rotatedCiY[d];
        MrotInv(7, d) = ciXsquare - ciYsquare;
        MrotInv(8, d) = rotatedCiX[d] * rotatedCiY[d];
    }

    MinvMrot = Minv * Mrot;
    MinvMrotInv = Minv * MrotInv;
}
