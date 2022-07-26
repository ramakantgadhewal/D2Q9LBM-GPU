#ifndef __CUDA_HELPER_CUH__
#define __CUDA_HELPER_CUH__

#include <fluid.hpp>
#include <stdio.h>

// #define cudaCheck(lastError) {                                              \
//     if (lastError != cudaSuccess) {                                         \
//         printf("Cuda failure %s:%d: '%s' '%s'\n", __FILE__, __LINE__,       \
//             cudaGetErrorName(lastError), cudaGetErrorString(lastError));    \
//     }                                                                       \
// }

#define cudaCheck(lastError) {                                              \
    if (lastError != cudaSuccess) {                                         \
        printf("Cuda failure %s:%d: '%s' '%s'\n", __FILE__, __LINE__,       \
            cudaGetErrorName(lastError), cudaGetErrorString(lastError));    \
        exit(-1);                                                           \
    }                                                                       \
}

#endif