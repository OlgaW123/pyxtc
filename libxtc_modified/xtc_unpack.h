/*
 * Copyright (c) 2020, Nikolay A. Krylov
 * All rights reserved.
 */
#include <stdint.h> //Available in MSVC 2010+, gcc clang

struct frame_data {
    int32_t minint[3];
    int32_t maxint[3];
    uint32_t smli;
    int32_t natoms;
    float inv_p;
    int32_t nt;
};

/*
 * Please note: input memory buffer size (packed_data) MUST be multiple of 8.
 * 8-byte pointer alignment of the input buffer is recommended also.
 */
extern "C" bool unpack_frame(const frame_data&fd, const uint64_t*packed_data, float *crds, int size);
