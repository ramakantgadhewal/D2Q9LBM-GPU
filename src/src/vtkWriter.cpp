#include <vtkWriter.hpp>

#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib> 
#include <assert.h>

using namespace std;

inline bool inRange(Fluid * fluid, int x, int y)
{
    if ((x >= 0) && (x <= fluid->nX - 1) && 
        (y >= 0) && (y <= fluid->nY - 1))
        return true;
    else
        return false;
}

inline int getIndexCpu(Fluid * fluid, int x, int y)
{
    return y * (fluid->nX) + x;
}

inline float calVecCrossXCpu(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1Y * vec2Z - vec1Z * vec2Y;
}

inline float calVecCrossYCpu(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1Z * vec2X - vec1X * vec2Z;
}

inline float calVecCrossZCpu(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1X * vec2Y - vec1Y * vec2X;
}

inline float deltaCpu(float x)
{
    if (x < 1) return 1 - x;
    else return 0;
}

// Find and output the maxium velocity of given fluid
void findMaxVel(Solver * solver, Fluid * fluid, string project, int time)
{
    float maxVel = 0;
    int maxVelIdX, maxVelIdY;

    for (int y = 0; y < fluid->nY; y++) {
        for (int x = 0; x < fluid->nX; x++) {
            int idx = getIndexCpu(fluid, x, y);
            float curVel = sqrt(pow(fluid->vel2[2 * idx], 2) + pow(fluid->vel2[2 * idx + 1], 2));
            if (curVel > maxVel) {
                maxVel = curVel;
                maxVelIdX = x;
                maxVelIdY = y;
            }
        }
    }

    cout << "maxVel: " << maxVel << " Idx: " << maxVelIdX << " " << maxVelIdY << " " << "\n";
}

// Write raw data of moving fluid
void writeFluidDataVtk(Solver * solver, Fluid * fluid, string project, int time) 
{
    stringstream outputFilename;
    outputFilename << "../../data/" << project.c_str() << "/vtkData/fluid_" << time << ".vtk";
    ofstream outputFile;

    outputFile.open(outputFilename.str().c_str());

    // VTK file headers
    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "fluid_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET RECTILINEAR_GRID\n";
    outputFile << "DIMENSIONS " << fluid->nX << " " << fluid->nY << " 1" << "\n";    
    outputFile << "X_COORDINATES " << fluid->nX << " float\n";

    for(int x = 0; x < fluid->nX; ++x) {
        outputFile << x << " ";
    }

    outputFile << "\n";
    outputFile << "Y_COORDINATES " << fluid->nY << " float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        outputFile << y << " ";
    }

    outputFile << "\n";
    outputFile << "Z_COORDINATES " << 1 << " float\n";
    outputFile << 0 << "\n";
    outputFile << "POINT_DATA " << fluid->nX * fluid->nY << "\n";

    // Write velocity
    outputFile << "VECTORS velocity_vector float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        for(int x = 0; x < fluid->nX; ++x) {
            int idx = y * (fluid->nX) + x;
            outputFile << fluid->vel2[2 * idx] << " " << fluid->vel2[2 * idx + 1] << " " << "0\n";
        }
    }

    // Close file
    outputFile.close();

    return;
}

// Write the Status of given fluid
void writeFluidStatusVtk(Solver * solver, Fluid * fluid, string project, int time) 
{
    stringstream outputFilename;
    outputFilename << "../../../data/" << project.c_str() << "/vtkData/status_" << time << ".vtk";
    ofstream outputFile;

    outputFile.open(outputFilename.str().c_str());

    // VTK file headers
    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "fluid_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET RECTILINEAR_GRID\n";
    outputFile << "DIMENSIONS " << fluid->nX << " " << fluid->nY << " 1" << "\n";    
    outputFile << "X_COORDINATES " << fluid->nX << " float\n";

    for(int x = 0; x < fluid->nX; ++x) {
        outputFile << x << " ";
    }

    outputFile << "\n";
    outputFile << "Y_COORDINATES " << fluid->nY << " float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        outputFile << y << " ";
    }

    outputFile << "\n";
    outputFile << "Z_COORDINATES " << 1 << " float\n";
    outputFile << 0 << "\n";
    outputFile << "POINT_DATA " << fluid->nX * fluid->nY << "\n";

    // Write velocity
    outputFile << "VECTORS velocity_vector float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        for(int x = 0; x < fluid->nX; ++x) {
            int idx = y * (fluid->nX) + x;
            if (fluid->status1[idx] == status_fluid) outputFile << "0 0 0" << "\n";
            else outputFile << "1 0 0" << "\n";
        }
    }

    // Close file
    outputFile.close();

    return;
}

// Write corrected data of moving fluid
void writeFluidDataStaticVtk(Solver * solver, Fluid * fluid, string project, int time) 
{
    stringstream outputFilename;
    outputFilename << "../../../data/" << project.c_str() << "/vtkDataStatic/fluid_" << time << ".vtk";
    ofstream outputFile;

    outputFile.open(outputFilename.str().c_str());

    // VTK file headers
    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "fluid_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET RECTILINEAR_GRID\n";
    outputFile << "DIMENSIONS " << fluid->nX << " " << fluid->nY << " 1" << "\n";    
    outputFile << "X_COORDINATES " << fluid->nX << " float\n";

    for(int x = 0; x < fluid->nX; ++x) {
        outputFile << x << " ";
    }

    outputFile << "\n";
    outputFile << "Y_COORDINATES " << fluid->nY << " float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        outputFile << y << " ";
    }

    outputFile << "\n";
    outputFile << "Z_COORDINATES " << 1 << " float\n";
    outputFile << 0 << "\n";
    outputFile << "POINT_DATA " << fluid->nX * fluid->nY << "\n";

    // Write velocity
    outputFile << "VECTORS velocity_vector float\n";

    for(int y = 0; y < fluid->nY; ++y) {
        for(int x = 0; x < fluid->nX; ++x) {
            int idx = y * (fluid->nX) + x;

            float posInFineCoorX = x - fluid->nX / 2.0;
            float posInFineCoorY = y - fluid->nY / 2.0;

            float gridVelX = fluid->velX + calVecCrossXCpu(0, 0, fluid->angVelZ, posInFineCoorX, posInFineCoorY, 0); 
            float gridVelY = fluid->velY + calVecCrossYCpu(0, 0, fluid->angVelZ, posInFineCoorX, posInFineCoorY, 0); 

            float uxAbsNotRotated = fluid->vel2[2 * idx    ] + gridVelX;
            float uyAbsNotRotated = fluid->vel2[2 * idx + 1] + gridVelY;

            float uxAbsRotated = fluid->rotMatrixInv[0] * uxAbsNotRotated + fluid->rotMatrixInv[1] * uyAbsNotRotated;
            float uyAbsRotated = fluid->rotMatrixInv[3] * uxAbsNotRotated + fluid->rotMatrixInv[4] * uyAbsNotRotated;

            outputFile << uxAbsRotated << " " << uyAbsRotated << " " << "0\n";
        }
    }

    // Close file
    outputFile.close();

    return;
}

// Write integrated data of whole fluid
void writeFluidDataIntegratedVtk(Solver ** solverList, Fluid ** fluidList, string project, int time) 
{
    /////////////////////////////////////////
    // ---------- VTK file headers ----------
    /////////////////////////////////////////
    stringstream outputFilename;
    outputFilename << "../../data/" << project.c_str() << "/vtkDataIntegrated/fluid_" << time << ".vtk";
    std::cout << "../../data/" << project.c_str() << "/vtkDataIntegrated/fluid_" << time << ".vtk" << "\n";
    ofstream outputFile;

    outputFile.open(outputFilename.str().c_str());
    
    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "fluid_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET RECTILINEAR_GRID\n";
    outputFile << "DIMENSIONS " << fluidList[0]->nX << " " << fluidList[0]->nY << " 1" << "\n";    
    outputFile << "X_COORDINATES " << fluidList[0]->nX << " float\n";

    for(int x = 0; x < fluidList[0]->nX; ++x) {
        outputFile << x << " ";
    }

    outputFile << "\n";
    outputFile << "Y_COORDINATES " << fluidList[0]->nY << " float\n";

    for(int y = 0; y < fluidList[0]->nY; ++y) {
        outputFile << y << " ";
    }

    outputFile << "\n";
    outputFile << "Z_COORDINATES " << 1 << " float\n";
    outputFile << 0 << "\n";
    outputFile << "POINT_DATA " << fluidList[0]->nX * fluidList[0]->nY << "\n";

    /////////////////////////////////////////
    // ---------- Write Velocity ------------
    /////////////////////////////////////////
    outputFile << "VECTORS velocity_vector float\n";

    for(int y = 0; y < fluidList[0]->nY; ++y) {
        for(int x = 0; x < fluidList[0]->nX; ++x) {
            int idx = y * (fluidList[0]->nX) + x;

            if (fluidList[0]->status1[idx] == status_invalid) {
                Fluid * fF = fluidList[1];

                float velXNoRot = 0;
                float velYNoRot = 0;

                float posInFineCoorX = fF->scale * (fF->rotMatrixInv[0] * (x - fF->posX) + fF->rotMatrixInv[1] * (y - fF->posY));
                float posInFineCoorY = fF->scale * (fF->rotMatrixInv[3] * (x - fF->posX) + fF->rotMatrixInv[4] * (y - fF->posY));

                // Idx of Bottom West South node
                int BWSidxX = (int) (posInFineCoorX + fF->nX / 2.0f);
                int BWSidxY = (int) (posInFineCoorY + fF->nY / 2.0f);
            
                for (int fY = BWSidxY; fY <= BWSidxY + 1; fY++) {
                    for (int fX = BWSidxX; fX <= BWSidxX + 1; fX++) {
                        int neighborIdx = getIndexCpu(fF, fX, fY);

                        float currNeiPosInFineCoorX = fX - fF->nX / 2.0f;
                        float currNeiPosInFineCoorY = fY - fF->nY / 2.0f;

                        float distX = abs(posInFineCoorX - currNeiPosInFineCoorX);
                        float distY = abs(posInFineCoorY - currNeiPosInFineCoorY);

                        float weight = deltaCpu(distX) * deltaCpu(distY);
                    
                        velXNoRot += fF->vel2[2 * neighborIdx]     * weight;
                        velYNoRot += fF->vel2[2 * neighborIdx + 1] * weight;
                    }
                }

                float velXTemp = velXNoRot + calVecCrossXCpu(0, 0, fF->angVelZ, posInFineCoorX, posInFineCoorY, 0);
                float velYTemp = velYNoRot + calVecCrossYCpu(0, 0, fF->angVelZ, posInFineCoorX, posInFineCoorY, 0);

                fluidList[0]->vel2[2 * idx    ] = fF->velX + fF->rotMatrix[0] * velXTemp + fF->rotMatrix[1] * velYTemp;
                fluidList[0]->vel2[2 * idx + 1] = fF->velY + fF->rotMatrix[3] * velXTemp + fF->rotMatrix[4] * velYTemp;

            }

            outputFile << fluidList[0]->vel2[2 * idx] << " " << fluidList[0]->vel2[2 * idx + 1] << " 0\n";
        }
    }

    outputFile.close();
}

// Write integrated data of whole fluid
void writeFluidDataIntegratedInFineResolutionVtk(Solver ** solverList, Fluid ** fluidList, string project, int time)
{
    /////////////////////////////////////////
    // -------- Determine data size ---------
    /////////////////////////////////////////
    int s = fluidList[1]->scale;

    int largeNX = (fluidList[0]->nX - 1) * fluidList[1]->scale;
    int largeNY = (fluidList[0]->nY - 1) * fluidList[1]->scale;

    /////////////////////////////////////////
    // ---------- VTK file headers ----------
    /////////////////////////////////////////
    stringstream outputFilename;
    outputFilename << "../../../data/" << project.c_str() << "/vtkDataIntegratedInFineResolution/fluid_" << time << ".vtk";
    ofstream outputFile;

    outputFile.open(outputFilename.str().c_str());
    
    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "fluid_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET RECTILINEAR_GRID\n";
    outputFile << "DIMENSIONS " << largeNX << " " << largeNY << " 1" << "\n";    
    outputFile << "X_COORDINATES " << largeNX << " float\n";

    for(int x = 0; x < largeNX; ++x) {
        outputFile << x << " ";
    }

    outputFile << "\n";
    outputFile << "Y_COORDINATES " << largeNY << " float\n";

    for(int y = 0; y < largeNY; ++y) {
        outputFile << y << " ";
    }

    outputFile << "\n";
    outputFile << "Z_COORDINATES " << 1 << " float\n";
    outputFile << 0 << "\n";
    outputFile << "POINT_DATA " << largeNX * largeNY << "\n";

    /////////////////////////////////////////
    // ---------- Write Velocity ------------
    /////////////////////////////////////////
    outputFile << "VECTORS velocity_vector float\n";

    for(int y = 0; y < largeNY; ++y) {
        for(int x = 0; x < largeNX; ++x) {

            // Determine neighbor nodes in coarse grids
            // Interpolate from coarse grid if all the 8 nodes nearby are valid, 
            // otherwise interpolate from fine grid
            float xInCoarse = (float) x / (float) s;
            float yInCoarse = (float) y / (float) s;

            int bws_coarse_x = (int) floor(xInCoarse);
            int bws_coarse_y = (int) floor(yInCoarse);

            bool inter_from_coarse = true;
            for(int cY = bws_coarse_y; cY <= bws_coarse_y + 1; cY++) {
                for(int cX = bws_coarse_x; cX <= bws_coarse_x + 1; cX++) {
                    int idx = getIndexCpu(fluidList[0], cX, cY);
                    if (fluidList[0]->status1[idx] == status_invalid) {
                        inter_from_coarse = false;
                    }
                }
            }

            float velX = 0, velY = 0;

            // Interpolate from coarse nodes
            if (inter_from_coarse == true) {
                for(int cY = bws_coarse_y; cY <= bws_coarse_y + 1; cY++) {
                    for(int cX = bws_coarse_x; cX <= bws_coarse_x + 1; cX++) {
                        int neighborIdx = getIndexCpu(fluidList[0], cX, cY);
                        assert(inRange(fluidList[0], cX, cY));

                        float distX = abs(xInCoarse - cX);
                        float distY = abs(yInCoarse - cY);

                        float weight = deltaCpu(distX) * deltaCpu(distY);
                    
                        velX += fluidList[0]->vel2[2 * neighborIdx]     * weight;
                        velY += fluidList[0]->vel2[2 * neighborIdx + 1] * weight;
                    }
                }
                outputFile << velX << " " << velY << " " << "0\n";
            }

            // Interpolate from fine nodes
            else {
                Fluid * fF = fluidList[1];

                float velXNoRot = 0;
                float velYNoRot = 0;

                float posInFineCoorX = fF->scale * (fF->rotMatrixInv[0] * (xInCoarse - fF->posX) + fF->rotMatrixInv[1] * (yInCoarse - fF->posY));
                float posInFineCoorY = fF->scale * (fF->rotMatrixInv[3] * (xInCoarse - fF->posX) + fF->rotMatrixInv[4] * (yInCoarse - fF->posY));

                // Idx of Bottom West South node
                int BWSidxX = (int) (posInFineCoorX + fF->nX / 2.0f);
                int BWSidxY = (int) (posInFineCoorY + fF->nY / 2.0f);
                
                for (int fY = BWSidxY; fY <= BWSidxY + 1; fY++) {
                    for (int fX = BWSidxX; fX <= BWSidxX + 1; fX++) {
                        int neighborIdx = getIndexCpu(fF, fX, fY);

                        float currNeiPosInFineCoorX = fX - fF->nX / 2.0f;
                        float currNeiPosInFineCoorY = fY - fF->nY / 2.0f;

                        float distX = abs(posInFineCoorX - currNeiPosInFineCoorX);
                        float distY = abs(posInFineCoorY - currNeiPosInFineCoorY);

                        float weight = deltaCpu(distX) * deltaCpu(distY);
                    
                        velXNoRot += fF->vel2[2 * neighborIdx]     * weight;
                        velYNoRot += fF->vel2[2 * neighborIdx + 1] * weight;
                    }
                }

                float velXTemp = velXNoRot + calVecCrossXCpu(0, 0, fF->angVelZ, posInFineCoorX, posInFineCoorY, 0);
                float velYTemp = velYNoRot + calVecCrossYCpu(0, 0, fF->angVelZ, posInFineCoorX, posInFineCoorY, 0);

                velX = fF->velX + fF->rotMatrix[0] * velXTemp + fF->rotMatrix[1] * velYTemp;
                velY = fF->velY + fF->rotMatrix[3] * velXTemp + fF->rotMatrix[4] * velYTemp;

                outputFile << velX << " " << velY << " " << "0\n";
            }
        }
    }

    outputFile.close();

    return;
}

// Write the 2D invalid position
void writeParticleTraceData(Solver ** solverList, Fluid ** fluidList, string project, int time)
{
    /////////////////////////////////////////
    // ---------- VTK file headers ----------
    /////////////////////////////////////////
    stringstream outputFilename; 
    outputFilename << "../../../data/" << project.c_str() << "/vtkDataIntegrated/boundary_" << time << ".vtk";
    ofstream outputFile; 
    outputFile.open(outputFilename.str().c_str());

    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "particle_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET POLYDATA\n";

    /////////////////////////////////////////////
    // ----- Determine Vertices coordinates -----
    /////////////////////////////////////////////
    outputFile << "POINTS " << 4 << " float\n";

    for (int i = 1; i < 2; i++) {
        Fluid * fF = fluidList[i];

        float posXRot, posYRot, posZRot;
        float s = fluidList[1]->scale;
        float posXList[4] = {fF->invalidSizeX / 2.0f,   fF->invalidSizeX / 2.0f, - fF->invalidSizeX / 2.0f, - fF->invalidSizeX / 2.0f};
        float posYList[4] = {fF->invalidSizeY / 2.0f, - fF->invalidSizeY / 2.0f, - fF->invalidSizeY / 2.0f,   fF->invalidSizeY / 2.0f};

        for (int j = 0; j < 4; j++) {
            posXRot = fF->rotMatrix[0] * posXList[j] + fF->rotMatrix[1] * posYList[j] + fF->posX;
            posYRot = fF->rotMatrix[3] * posXList[j] + fF->rotMatrix[4] * posYList[j] + fF->posY;
            outputFile << posXRot << " " << posYRot << " 0\n";
        }
    }

    ///////////////////////////////////////
    // ---------- Write Vertices ----------
    ///////////////////////////////////////
    outputFile << "VERTICES 1 " << 4 + 1 << "\n";
    outputFile << 4 << " ";

    for(int n = 0; n < 4; ++n) {
        outputFile << n << " ";
    }

    ////////////////////////////////////////////////////
    // ----- Write lines between neighboring nodes -----
    ////////////////////////////////////////////////////
    outputFile << "LINES " << 4 << " " << 3 * 4 << "\n";

    for(int n = 0; n < 4; ++n) {
        outputFile << "2 " << n << " " << (n + 1) % 4 << "\n";
    }    

    outputFile.close();
    return;
}

// Write the 2D invalid position
void writeParticleTraceDataInFineResolution(Solver ** solverList, Fluid ** fluidList, string project, int time)
{
    /////////////////////////////////////////
    // ---------- VTK file headers ----------
    /////////////////////////////////////////
    stringstream outputFilename; 
    outputFilename << "../../../data/" << project.c_str() << "/vtkDataIntegratedInFineResolution/boundary_" << time << ".vtk";
    ofstream outputFile; 
    outputFile.open(outputFilename.str().c_str());

    outputFile << "# vtk DataFile Version 3.0\n";
    outputFile << "particle_state\n";
    outputFile << "ASCII\n";
    outputFile << "DATASET POLYDATA\n";

    /////////////////////////////////////////////
    // ----- Determine Vertices coordinates -----
    /////////////////////////////////////////////
    outputFile << "POINTS " << 4 << " float\n";

    for (int i = 1; i < 2; i++) {
        Fluid * fF = fluidList[i];

        float posXRot, posYRot, posZRot;
        float s = fluidList[1]->scale;
        float posXList[4] = {fF->invalidSizeX / 2.0f * s,   fF->invalidSizeX / 2.0f * s, - fF->invalidSizeX / 2.0f * s, - fF->invalidSizeX / 2.0f * s};
        float posYList[4] = {fF->invalidSizeY / 2.0f * s, - fF->invalidSizeY / 2.0f * s, - fF->invalidSizeY / 2.0f * s,   fF->invalidSizeY / 2.0f * s};

        for (int j = 0; j < 4; j++) {
            posXRot = fF->rotMatrix[0] * posXList[j] + fF->rotMatrix[1] * posYList[j] + fF->posX * s;
            posYRot = fF->rotMatrix[3] * posXList[j] + fF->rotMatrix[4] * posYList[j] + fF->posY * s;
            outputFile << posXRot << " " << posYRot << " 0\n";
        }
    }

    ///////////////////////////////////////
    // ---------- Write Vertices ----------
    ///////////////////////////////////////
    outputFile << "VERTICES 1 " << 4 + 1 << "\n";
    outputFile << 4 << " ";

    for(int n = 0; n < 4; ++n) {
        outputFile << n << " ";
    }

    ////////////////////////////////////////////////////
    // ----- Write lines between neighboring nodes -----
    ////////////////////////////////////////////////////
    outputFile << "LINES " << 4 << " " << 3 * 4 << "\n";

    for(int n = 0; n < 4; ++n) {
        outputFile << "2 " << n << " " << (n + 1) % 4 << "\n";
    }    

    outputFile.close();
    return;
}