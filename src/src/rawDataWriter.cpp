#include <rawDataWriter.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

void writeRawData(int time, Solver * solver, Fluid * fluid)
{
    string fileName = "../data/rawData/raw" + to_string(time) + ".bin";
    FILE * file = fopen(fileName.c_str(), "w");

    fwrite(fluid->fNewDim, sizeof(float), dim * fluid->nAll, file);
    fclose(file);
}

void readRawData(int time, Solver * solver, Fluid * fluid)
{
    string fileName = "../data/rawData/raw" + to_string(time) + ".bin";
    FILE * file = fopen(fileName.c_str(), "r");

    fread(fluid->fNewDim, sizeof(float), dim * fluid->nAll, file);
    fclose(file);
}
