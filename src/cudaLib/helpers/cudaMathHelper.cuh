#ifndef __CUDA_MATH_HELPER_CUH__
#define __CUDA_MATH_HELPER_CUH__

#include <fluid.hpp>
#include <stdio.h>

__device__ inline bool inRange(Fluid * fluid, int x, int y)
{
    if ((x >= 0) && (x <= fluid->nX - 1) && 
        (y >= 0) && (y <= fluid->nY - 1))
        return true;
    else
        return false;
}

__device__ inline bool isBoundary(Fluid * fluid, int x, int y)
{
    if (inRange(fluid, x, y)) {
        if ((x == 0) || (x == fluid->nX - 1) || 
            (y == 0) || (y == fluid->nY - 1)) {
            return true;
        }
    }
    return false;
}


__device__ inline int getIndex(Fluid * fluid, int x, int y)
{
    return y * (fluid->nX) + x;
}

__device__ inline float calVecCrossX(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1Y * vec2Z - vec1Z * vec2Y;
}

__device__ inline float calVecCrossY(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1Z * vec2X - vec1X * vec2Z;
}

__device__ inline float calVecCrossZ(float vec1X, float vec1Y, float vec1Z, float vec2X, float vec2Y, float vec2Z)
{
    return vec1X * vec2Y - vec1Y * vec2X;
}

__device__ inline float delta(float x)
{
    if (x < 1) return 1 - abs(x);
    else return 0;
}

#endif