#ifndef __RAW_DATA_WRITER_HPP__
#define __RAW_DATA_WRITER_HPP__

#include <solver.hpp>
#include <fluid.hpp>

void writeRawData(int time, Solver * solver, Fluid * fluid);
void readRawData(int time, Solver * solver, Fluid * fluid);

#endif