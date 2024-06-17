/*
 * Copyright (c) 2009-2014, Erik Lindahl & David van der Spoel
 * Copyright (c) 2016-2020, Nikolay A. Krylov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "xtc_unpack.h"
#include <cstdlib>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined _WIN32 || defined __CYGWIN__
#define DLL_PUBLIC __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#else
#define DLL_PUBLIC
#endif
#endif

/*
 * Endian handling ideas and routines were adopted from:
 * https://gist.github.com/panzi/6856583
 * https://github.com/blizzard4591/cmake-portable-endian
 * https://github.com/Tencent/rapidjson/commit/2d732794f0057d0bafd3406ebaf868a55942b5ab
 *
 */

#ifdef __GLIBC__
#include <byteswap.h>


#define bswap32 bswap_32
#define bswap64 bswap_64

//Compiler-specific variants
#elif defined(_MSC_VER) && _MSC_VER >= 1300
#include <stdlib.h>
#define bswap32 _byteswap_ulong
#define bswap64 _byteswap_uint64
#elif defined(__clang__)
#if __has_builtin(__builtin_bswap32)
#define bswap32 __builtin_bswap32
#endif
#if __has_builtin(__builtin_bswap64)
#define bswap64 __builtin_bswap64
#endif
#elif defined(__GNUC__)  // Supported since at least GCC 4.4
#define bswap32 __builtin_bswap32
#define bswap64 __builtin_bswap64
#else

//default fallback
uint32_t bswap32(uint32_t v) {
    return ((((uint32_t) (v) << 24))
            | (((uint32_t) (v) << 8) & uint32_t(0x00FF0000))
            | (((uint32_t) (v) >> 8) & uint32_t(0x0000FF00))
            | (((uint32_t) (v) >> 24)));
}

uint64_t bswap64(uint64_t v) {
    return ((((uint64_t) (v) << 56))
            | (((uint64_t) (v) << 40) & uint64_t(0x00FF000000000000))
            | (((uint64_t) (v) << 24) & uint64_t(0x0000FF0000000000))
            | (((uint64_t) (v) << 8) & uint64_t(0x000000FF00000000))
            | (((uint64_t) (v) >> 8) & uint64_t(0x00000000FF000000))
            | (((uint64_t) (v) >> 24) & uint64_t(0x0000000000FF0000))
            | (((uint64_t) (v) >> 40) & uint64_t(0x000000000000FF00))
            | (((uint64_t) (v) >> 56)));
}
#endif

//Adapted from rapidjson.h
#define CPU_LITTLEENDIAN 0
#define CPU_BIGENDIAN 1

#ifdef __BYTE_ORDER__
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define CPU_ENDIANNESS CPU_LITTLEENDIAN
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define CPU_ENDIANNESS CPU_BIGENDIAN
#endif

// Detect with GLIBC's endian.h
#elif defined(__GLIBC__)
#include <endian.h>
#if (__BYTE_ORDER == __LITTLE_ENDIAN)
#define CPU_ENDIANNESS CPU_LITTLEENDIAN
#elif (__BYTE_ORDER == __BIG_ENDIAN)
#define CPU_ENDIANNESS CPU_BIGENDIAN
#endif // __GLIBC__
#endif // __BYTE_ORDER__

struct work_data {
    const frame_data&fd;
    const uint32_t*packed_data;
    float *crds;
    int tid, nth;
};

namespace {

    /* Internal support routines for reading/writing compressed coordinates from xdrfile
     *
     * sizeofint - calculate smallest number of bits necessary
     * to represent a certain integer.
     */
    int
    sizeofint(unsigned int size) {
        unsigned int num = 1;
        unsigned int num_of_bits = 0;

        while (size >= num && num_of_bits < 32) {
            num_of_bits++;
            num <<= 1;
        }
        return num_of_bits;
    }

    /*
     * sizeofints - calculate 'bitsize' of compressed ints
     *
     * given a number of small unsigned integers and the maximum value
     * return the number of bits needed to read or write them with the
     * routines encodeints/decodeints. You need this parameter when
     * calling those routines.
     * (However, in some cases we can just use the variable 'smallidx'
     * which is the exact number of bits, and them we dont need to call
     * this routine).
     */
    int
    sizeofints(int num_of_ints, unsigned int sizes[]) {
        int i, num;
        unsigned int num_of_bytes, num_of_bits, bytes[32], bytecnt, tmp;
        num_of_bytes = 1;
        bytes[0] = 1;
        num_of_bits = 0;
        for (i = 0; i < num_of_ints; i++) {
            tmp = 0;
            for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
                tmp = bytes[bytecnt] * sizes[i] + tmp;
                bytes[bytecnt] = tmp & 0xff;
                tmp >>= 8;
            }
            while (tmp != 0) {
                bytes[bytecnt++] = tmp & 0xff;
                tmp >>= 8;
            }
            num_of_bytes = bytecnt;
        }
        num = 1;
        num_of_bytes--;
        while (bytes[num_of_bytes] >= num) {
            num_of_bits++;
            num *= 2;
        }
        return num_of_bits + num_of_bytes * 8;

    }

    static const int magicints[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
        80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
        1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
        16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
        131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
        832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
        4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216
    };

    const int FIRSTIDX = 9;
    /* note that magicints[FIRSTIDX-1] == 0 */
    const int LASTIDX = (sizeof (magicints) / sizeof (*magicints));

    inline bool is_little_endian() {
#ifdef CPU_ENDIANNESS
        return CPU_ENDIANNESS == CPU_LITTLEENDIAN;
#else

        union {
            uint16_t value;
            uint8_t data[sizeof (uint16_t)];
        } number;
        number.value = 1;
        return number.data[0];
#endif
    }

    template<typename T >
    inline void swapbytes(const T in, T&out) {
        const int tsize = sizeof (T);
        static_assert(tsize >= 4 && tsize <= 8, "T type not supported!");


        switch (tsize) {
            case 4:
            {
                reinterpret_cast<uint32_t &> (out) = bswap32(reinterpret_cast<uint32_t const &> (in));
                break;
            }
            case 8:
            {
                reinterpret_cast<uint64_t &> (out) = bswap64(reinterpret_cast<uint64_t const &> (in));
                break;
            }
        }
    }

    class bit_reader {
    public:

        bit_reader(const uint64_t* d, int s) : data(d), data32(reinterpret_cast<const uint32_t*>(d)), size32(s / 4) {
            if (s % 4 != 0) {
                size32 += 1;
            }
            init(0);
        }
        void init(uint64_t bitpos) {
            auto ibuf = bitpos / bit64;
            buf = data[ibuf];

            if (little_endian) {
                swapbytes(buf, buf);
            }
            curbit = ibuf*bit64;
            uint16_t bitshift = bitpos - curbit;

            next4b = 2 * (ibuf + 1);

            bitsavalable = 64;
            if (bitshift) {
                skip_load(bitshift);
            }
        }

        void skip(uint32_t len) {
            if (bitsavalable > len + 8) {
                skip_load(len);
            } else {
                init(curbit + len);
            }
        }

        //perform short read (1,5,8 bits)

        uint8_t read(uint8_t len) {
            auto tmp = buf;
            tmp >>= (bit64 - len);

            skip(len);
            return tmp;
        }

        //read up to 32 bits

        int read_int(int num_of_bits) {
            auto mask = (1 << num_of_bits) - 1;
            int ret = 0;
            //we should always have extra space, so extra read at the end of array is not a problem.
            auto num_of_bytes = std::min((num_of_bits >> 3) + 1, 4);
            for (int i = 0; i < num_of_bytes; i++) {
                ret |= ((int) read(8)) << (8 * i);
            }
            return ret&mask;
        }

        template <typename T>
        void unpack_from_int(int fullbytes, int partbits, const int32_t sizeint[], int32_t intcrds[]) {
            T v=0;
            int i = 0;
            for (; i < fullbytes; i++) {
                auto ibyte = ((T) read(8));
                v |= ibyte << (8 * i);
            }

            if (partbits) {
                v |= ((T) read(partbits)) << (8 * i);
            }

            uint_fast64_t sz = uint_fast64_t(sizeint[2]);
            uint_fast32_t sy = sizeint[1];
            uint_fast64_t szy = sz*sy;
            uint_fast32_t x1 = v / szy;
            T q1 = v - x1*szy;
            uint_fast32_t y1 = q1 / sz;
            uint_fast32_t z1 = q1 - y1*sz;

            intcrds[0] = x1;
            intcrds[1] = y1;
            intcrds[2] = z1;
        }

        void unpack(int bitsize, const int32_t sizeint[], int32_t intcrds[]) {


            auto fullbytes = bitsize >> 3;
            auto partbits = bitsize & ((1 << 3) - 1);

            if (bitsize <= 64) {
                unpack_from_int<uint64_t>(fullbytes, partbits, sizeint, intcrds);
            } else

#ifdef HAS_INT128
            {
                typedef unsigned __int128 uint128;
                unpack_from_int<uint128>(fullbytes, partbits, sizeint, intcrds);
            }
#else
                {
                    int bytes[16];

                    int nbytes = 0;
                    for (; nbytes < fullbytes; nbytes++) {
                        bytes[nbytes] = read(8);
                    }

                    if (partbits) {
                        bytes[nbytes++] = read(partbits);
                    }

                    int i, j, p, num;
                    static const int num_of_ints = 3;

                    for (i = num_of_ints - 1; i > 0; i--) {
                        num = 0;
                        for (j = nbytes - 1; j >= 0; j--) {
                            num = (num << 8) | bytes[j];
                            p = num / sizeint[i];
                            bytes[j] = p;
                            num = num - p * sizeint[i];
                        }
                        intcrds[i] = num;
                    }
                    intcrds[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
                }
#endif
        }

    private:

        void skip_load(uint32_t len) {
            buf <<= len;
            bitsavalable -= len;
            curbit += len;
            if (bitsavalable < bit32) {
                uint32_t next32 = 0;
                if(next4b < size32){
                next32 = data32[next4b];
                }
                next4b++;
                if (little_endian) {
                    swapbytes(next32, next32);
                }
                buf |= uint64_t(next32) << (bit32 - bitsavalable);
                bitsavalable += bit32;
            }
        }

        static const int bit32 = 32;
        static const int bit64 = 64;
        uint64_t buf, next4b, curbit;
        const uint64_t*data;
        const uint32_t*data32;
        int size32;
        uint16_t bitsavalable;
        bool little_endian = is_little_endian(); //ppc issue!
    };

    struct v3i {
        static const int ND = 3;
        static float inv_p;
        int32_t V[ND];
        v3i(int v = 0) : V{v, v, v} {}
        template <typename T>
        v3i(const T*v) : V{int32_t(v[0]), int32_t(v[1]), int32_t(v[2])}{}

        inline v3i & operator+=(const v3i&u) {
            for (int i = 0; i < ND; i++) {
                V[i] += u.V[i];
            }
            return *this;
        };

        inline v3i & operator-=(const v3i&u) {
            for (int i = 0; i < ND; i++) {
                V[i] -= u.V[i];
            }
            return *this;
        };

        inline void flt_convert(float*v) {
            for (int i = 0; i < v3i::ND; i++) {
                v[i] = inv_p * V[i];
            }
        }
    };

    float v3i::inv_p=1;
}

extern "C" DLL_PUBLIC
bool unpack_frame(const frame_data&fd, const uint64_t*packed_data, float *crds, int size) {
    auto smlim = fd.smli - 1;
    smlim = (FIRSTIDX > smlim) ? FIRSTIDX : smlim;

    uint32_t sizeint[3];
    unsigned int bitsize;
    unsigned bitsizeint[3];
    for (int i = 0; i < 3; i++) {
        sizeint[i] = fd.maxint[i] - fd.minint[i] + 1;
    }
    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        bitsize = 0; /* flag the use of large sizes */
    } else {
        bitsize = sizeofints(3, sizeint);
    }
    auto large = 0 == bitsize;

    v3i::inv_p=fd.inv_p;

    bool ret_ok = true;
#ifdef _OPENMP
    auto ntuser = std::min(fd.nt, omp_get_num_procs());
    auto na = fd.natoms;
    int blk_sz = std::max(1, std::min(512, int (float(na) / ntuser / 5)));

#pragma omp parallel num_threads(ntuser) if(na>500)
#endif
    {

#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        static const int tid = 0;
        static const int nth = 1;
        static const int blk_sz = 1;
#endif

        bit_reader br(packed_data, size);

        unsigned int smallidx = fd.smli;
        int smaller = magicints[smlim] / 2;
        int smallnum = magicints[smallidx] / 2;

        int i = 0, is_smaller;
        bool flag;
        uint8_t run = 0;

        v3i thiscrds, prevcoord, thissmallcrds, vminint(fd.minint), vsizeint(sizeint);
        auto ssmall = magicints[smallidx];

        int maincntr = 0;

        while (i < fd.natoms) {

            auto *crd3f = crds + 3 * i;
            auto write = 0 == (((maincntr / blk_sz) + tid) % nth);

            if (!large) {
                if (write) {
                    br.unpack(bitsize, vsizeint.V, thiscrds.V);
                } else {
                    br.skip(bitsize);
                }
            } else {
                for (int ibig = 0; ibig < 3; ibig++) {
                    if (write) {
                        thiscrds.V[ibig] = br.read_int(bitsizeint[ibig]);
                    } else {
                        br.skip(bitsizeint[ibig]);
                    }
                }
            }
            thiscrds += vminint;
            prevcoord = thiscrds;

            i++;

            flag = br.read(1);
            is_smaller = 0;
            if (flag) {
                run = br.read(5);
                is_smaller = run % 3;
                run -= is_smaller;
                is_smaller--;
            }

            if (run > 0) {
                // write rle-encoded coords

                v3i vsmallnum(smallnum);
                v3i szsmall3(ssmall);

                //prevent crash due to damaged frame (3064-5-95-58.xtc)
                if (fd.natoms - i < run / 3) {
                    ret_ok = false;
                    break;
                }

                for (int k = 0, scntr = 0; (k < run) && (i < fd.natoms); k += 3, scntr++) {
                    if (write) {
                        br.unpack(smallidx, szsmall3.V, thissmallcrds.V);

                        auto tmp = prevcoord;
                        tmp -= vsmallnum;
                        thissmallcrds += tmp;

                        if (0 == k) {
                            std::swap(prevcoord, thissmallcrds);
                            prevcoord.flt_convert(crd3f);
                            crd3f += 3;
                        } else {
                            prevcoord = thissmallcrds;
                        }
                        thissmallcrds.flt_convert(crd3f);
                        crd3f += 3;
                    } else {
                        br.skip(smallidx);
                    }
                    i++;
                }
            } else {
                // write main coords only
                if (write) {
                    thiscrds.flt_convert(crd3f);
                    crd3f += 3;
                }
            }

            smallidx += is_smaller;

            if (is_smaller < 0) {
                smallnum = smaller;

                if (smallidx > FIRSTIDX) {
                    smaller = magicints[smallidx - 1] / 2;
                } else {
                    smaller = 0;
                }
            } else if (is_smaller > 0) {
                smaller = smallnum;
                smallnum = magicints[smallidx] / 2;
            }
            ssmall = magicints[smallidx];

            maincntr++;

        }
    }
    return ret_ok;
}
