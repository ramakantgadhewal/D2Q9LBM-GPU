#ifndef __USER_DEFINE_HPP__
#define __USER_DEFINE_HPP__

enum ColliModel 
{
    BGK,
    MRT,
};

// Zero: Simulation strats from init state
// File: Simulation strats from a .bin file
enum SimuStartPos
{
    zero,
    file,
};

const ColliModel colliModel = MRT;

const SimuStartPos simuStartPos = zero;

// CUDA
const int tileSize = 16;

const int dim = 9;

const float csSquare = 1. / 3.;
const float csSquareInv = 3;

// Overset Grid

#endif