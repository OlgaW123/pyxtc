'''*
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
 *'''
import os
import struct
import sys
import numpy as np

try:
    import xdrlib as xdrlib
except ImportError:
    import xdrlib3 as xdrlib


class frame_data:
    def __init__(self, minint=None, maxint=None, smli=0, natoms=0, inv_p=0.0, nt=0):
        if minint is None:
            minint = [0, 0, 0]  
        if maxint is None:
            maxint = [0, 0, 0] 

        self.minint = minint
        self.maxint = maxint
        self.smli = smli
        self.natoms = natoms
        self.inv_p = inv_p
        self.nt = nt


class XTCBadMagic(IOError):
    def __init__(self):
        super(XTCBadMagic, self).__init__("Bad frame magic!")


class XTCBadHeader(IOError):
    def __init__(self):
        super(XTCBadHeader, self).__init__("Bad frame header!")


class XTCBadFrame(IOError):
    pass


def bswap32(v):
    return ((v & 0xFF000000) >> 24) | ((v & 0x00FF0000) >> 8) | ((v & 0x0000FF00) << 8) | ((v & 0x000000FF) << 24)


def bswap64(v):
    return ((v & 0xFF00000000000000) >> 56) | ((v & 0x00FF000000000000) >> 40) | ((v & 0x0000FF0000000000) >> 24) | \
           ((v & 0x000000FF00000000) >> 8) | ((v & 0x00000000FF000000) << 8) | ((v & 0x0000000000FF0000) << 24) | \
           ((v & 0x000000000000FF00) << 40) | ((v & 0x00000000000000FF) << 56)


def get_endianness():
    return 'little' if sys.byteorder == 'little' else 'big'


def sizeofint(size):
    num = 1
    num_of_bits = 0
    while size >= num and num_of_bits < 32:
        num_of_bits += 1
        num <<= 1
    return num_of_bits


def sizeofints(num_of_ints, sizes):
    num_of_bytes = 1
    bytes = [1] + [0] * 31
    num_of_bits = 0
    for i in range(num_of_ints):
        tmp = 0
        bytecnt = 0
        while bytecnt < num_of_bytes:
            tmp = bytes[bytecnt] * sizes[i] + tmp
            bytes[bytecnt] = tmp & 0xff
            tmp >>= 8
            bytecnt += 1
        while tmp != 0:
            bytes[bytecnt] = tmp & 0xff
            tmp >>= 8
            bytecnt += 1
        num_of_bytes = bytecnt
    num = 1
    num_of_bytes -= 1
    while bytes[num_of_bytes] >= num:
        num_of_bits += 1
        num *= 2
    return num_of_bits + num_of_bytes * 8


magicints = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216
]
FIRSTIDX = 9
LASTIDX = len(magicints)


def is_little_endian():
    return sys.byteorder == 'little'


class BitReader:
    def __init__(self, data):
        self.data = data
        self.data32 = struct.unpack('<{}I'.format(len(data) * 2), data)
        self.curbit = 0
        self.bits_available = 0
        self.buf = 0
        self.next4b = 0
        self.little_endian = self.is_little_endian()
        self.init(0)

    def is_little_endian(self):
        return struct.unpack('<I', struct.pack('=I', 1))[0] == 1

    def swapbytes(self, data):
        if isinstance(data, np.ndarray):
            data = data.item()
        data = int(data)
        data_size = (data.bit_length() + 7) // 8
        if data_size <= 4:
            swapped = bswap32(data)
        else:
            swapped = bswap64(data)
        return np.uint64(swapped)

    def init(self, bitpos):
        ibuf = bitpos // 64
        self.buf = self.data[ibuf]
        if self.little_endian:
            self.buf = self.swapbytes(self.buf)
        self.curbit = ibuf * 64
        bitshift = bitpos - self.curbit
        self.next4b = 2 * (ibuf + 1)
        self.bits_available = 64
        if bitshift:
            self.skip_load(bitshift)

    def skip_load(self, leng):
        self.buf <<= np.uint32(leng)
        self.bits_available -= leng
        self.curbit += leng
        if self.bits_available < 32:
            if self.next4b < len(self.data32):
                next32 = self.data32[self.next4b]
            else:
                next32 = 0
            self.next4b += 1
            if self.little_endian:
                next32 = self.swapbytes(next32)
            self.buf |= np.uint64(next32) << np.uint64((32 - self.bits_available))
            self.bits_available += 32

    def skip(self, leng):
        if self.bits_available > leng + 8:
            self.skip_load(leng)
        else:
            self.init(self.curbit + leng)

    def read(self, leng):
        tmp = self.buf
        tmp >>= (np.uint64(64) - np.uint64(leng))
        self.skip(leng)
        return tmp

    def read_int(self, num_of_bits):
        mask = (1 << num_of_bits) - 1
        ret = 0
        num_of_bytes = min((num_of_bits >> 3) + 1, 4)
        for i in range(num_of_bytes):
            ret |= (int(self.read(8)) << (8 * i))
        return ret & mask

    def unpack_from_int(self, fullbytes, partbits, sizeint):
        v = np.uint64(0)
        for i in range(fullbytes):
            byte = self.read(8)
            byte <<= np.uint64(i * 8)
            v = v | byte
        if partbits:
            last_bits = self.read(partbits)
            last_bits <<= np.uint64(fullbytes * 8)
            v = v | last_bits
        sz = sizeint[2]
        sy = sizeint[1]
        szy = sz * sy
        x1 = v // szy
        q1 = v - x1 * szy
        y1 = q1 // sz
        z1 = q1 - y1 * sz
        intcrds = [0, 0, 0]
        intcrds[0] = x1
        intcrds[1] = y1
        intcrds[2] = z1
        return intcrds

    def unpack(self, bitsize, sizeint):
        fullbytes = bitsize >> 3
        partbits = bitsize & 7
        if bitsize <= 64:
            return self.unpack_from_int(fullbytes, partbits, sizeint)
        else:
            pass


class v3i:
    ND = 3
    inv_p = 1.0

    def __init__(self, v=0):
        if isinstance(v, (list, tuple, np.ndarray)):
            self.V = [int(v[0]), int(v[1]), int(v[2])]
        else:
            self.V = [v] * v3i.ND

    def __getitem__(self, index):
        return self.V[index]

    def __setitem__(self, index, value):
        self.V[index] = value

    def __iadd__(self, other):
        for i in range(v3i.ND):
            self.V[i] += other.V[i]
        return self

    def __isub__(self, other):
        for i in range(v3i.ND):
            self.V[i] -= other.V[i]
        return self

    def __sub__(self, other):
        result = v3i()
        for i in range(v3i.ND):
            result[i] = self.V[i] - other.V[i]
        return result

    def flt_convert(self):
        return [v3i.inv_p * self.V[i] for i in range(v3i.ND)]


def unpack_frame(fd, packed_data, crds):
    smlim = int(fd.smli - 1)
    smlim = max(FIRSTIDX, smlim)
    v3i.inv_p = fd.inv_p
    sizeint = [fd.maxint[i] - fd.minint[i] + 1 for i in range(3)]
    bitsizeint = [sizeofint(x) for x in sizeint]
    bitsize = 0 if any(x > 0xffffff for x in sizeint) else sizeofints(3, sizeint)
    large = bitsize == 0
    ret_ok = True
    br = BitReader(packed_data)
    smallidx = int(fd.smli)
    smaller = magicints[smlim] // 2
    smallnum = magicints[smallidx] // 2
    ssmall = magicints[smallidx]
    vminint = v3i(fd.minint)
    vsizeint = v3i(sizeint)
    i = 0
    maincntr = 0
    prevcoord = v3i(0)
    all_coords = []
    while i < fd.natoms:
        write = True
        thiscrds = v3i(0)
        if not large:
            if write:
                thiscrds = v3i(br.unpack(bitsize, vsizeint))
            else:
                br.skip(bitsize)
        else:
            for ibig in range(3):
                if write:
                    thiscrds[ibig] = br.read_int(bitsizeint[ibig])
                else:
                    br.skip(bitsizeint[ibig])
        thiscrds += vminint
        prevcoord = thiscrds
        i += 1
        flag = br.read(1)
        is_smaller = 0
        if flag:
            run = br.read(5)
            run = int(run)
            is_smaller = int(run % 3)
            run -= is_smaller
            is_smaller -= 1
        if run > 0:
            vsmallnum = v3i(smallnum)
            szsmall3 = v3i(ssmall)
            if fd.natoms - i < run // 3:
                ret_ok = False
                break
            for k in range(0, run, 3):
                thissmallcrds = v3i(0)
                if write:
                    thissmallcrds = v3i(br.unpack(int(smallidx), szsmall3))
                    tmp = prevcoord - vsmallnum
                    thissmallcrds += tmp
                    if k == 0:
                        prevcoord, thissmallcrds = thissmallcrds, prevcoord
                        all_coords.append(prevcoord.flt_convert())
                    else:
                        prevcoord = thissmallcrds
                    all_coords.append(thissmallcrds.flt_convert())
                else:
                    br.skip(int(smallidx))
                i += 1
        else:
            if write:
                all_coords.append(thiscrds.flt_convert())
        smallidx += is_smaller
        smallidx = int(smallidx)
        if is_smaller < 0:
            smallnum = smaller
            if smallidx > FIRSTIDX:
                smaller = magicints[smallidx - 1] // 2
            else:
                smaller = 0
        elif is_smaller > 0:
            smaller = smallnum
            smallnum = magicints[smallidx] // 2
        ssmall = magicints[smallidx]
        maincntr += 1
        
    return ret_ok, all_coords 


class XTCReader:
    mxtc = 1995
    mxtcbytes = struct.pack('>l', mxtc)
    hdr1 = 4 * 4 + 4 * 9 + 4
    hdr2 = 4 * 9
    hdrfull = hdr1 + hdr2
    flt3len = 3 * 4
    alignmentBytes = 1 << 2
    alignmentBytesMinusOne = alignmentBytes - 1

    @classmethod
    def _align4(cls, l):
        return (l + cls.alignmentBytesMinusOne) & ~cls.alignmentBytesMinusOne

    def __init__(self, fname, beg=-1, end=-1, dt=-1, nt=4):
        self.end = end
        self._frd = frame_data()
        self._frd.nt = nt
        self._buf = None
        self.X = None
        self.box = np.empty((3, 3), dtype=np.float32)
        self._fxtc = open(fname, "rb")
        self._fsize = os.path.getsize(fname)
        self.dt = -1
        self.iframe = 0
        self.next_frame(True, True)
        self.beg = self.t
        self._t0 = t0 = self.t
        self._dtrrj = 0
        self._teps = 0
        if self.next_frame(True, True):
            t1 = self.t
            self._dtrrj = dtrrj = t1 - t0
            self._teps = 1e-3 * dtrrj
        self._fxtc.seek(0)
        if beg > 0:
            self.beg = beg
            if self._dtrrj > 0:
                self._search_t(beg)
            else:
                self._skip_to(beg)
        else:
            self.t = self.beg
        if dt > 0:
            if dt < dtrrj:
                dt = -1
            else:
                scan_cost = 50000
                nfr_skip = int(dt / dtrrj)
                hdr_cost = scan_cost * self.hdrfull * nfr_skip
                datalen_cost = self.nacur * self.flt3len * 1.5 / 3
                self._bigdt = datalen_cost < hdr_cost
            self.dt = dt
        self.iframe = 0

    def _search_frame_hdr(self):
        start_pos = self._fxtc.tell()
        data = self._fxtc.read(int(1.5 * (self.hdrfull + self.datalen)))
        offset = 0
        while 1:
            magic_pos = data.find(self.mxtcbytes, offset)
            if magic_pos < 0:
                return False
            cur_pos = start_pos + magic_pos
            self._fxtc.seek(cur_pos)
            try:
                self._read_header(True)
                break
            except XTCBadHeader:
                pass
            offset += len(self.mxtcbytes)
        self._fxtc.seek(cur_pos)
        return True

    def _search_t(self, t):
        if self.eot():
            return
        off = int(max(0, (self.hdrfull + self.datalen) * ((t - self._t0) / self._dtrrj - 1.5)))
        if off >= self._fsize:
            self._fxtc.seek(0, os.SEEK_END)
            return
        self._fxtc.seek(off)
        if not self._search_frame_hdr():
            raise IOError("Failed to locate next frame header!")
        self._skip_to(t)

    def _skip_to(self, t):
        while not self.eot():
            frame_begin = self._fxtc.tell()
            self.next_frame(True, True)
            if self.t > t or np.abs(self.t - t) < self._teps:
                self._fxtc.seek(frame_begin)
                break

    def _read_header(self, check=False):
        mem = self._fxtc.read(self.hdr1)
        if not mem:
            return False
        xdrfile = xdrlib.Unpacker(mem)
        magic = xdrfile.unpack_int()
        if magic != self.mxtc:
            raise XTCBadMagic()
        self.nacur = xdrfile.unpack_uint()
        fr_idx = xdrfile.unpack_uint()
        self.t = xdrfile.unpack_float()
        for i in range(self.box.size):
            self.box[i // 3, i % 3] = xdrfile.unpack_float()
        self.lsize = xdrfile.unpack_uint()
        if check:
            if fr_idx < 0 or self.t < 0 or np.any(self.box[0][1:] > 0):
                raise XTCBadHeader()
        return True

    def _tnext(self):
        if self.dt > 0:
            return self.beg + self.iframe * self.dt
        else:
            return self.t + self._dtrrj

    def next_frame(self, hdr_only=False, no_check=False):
        if not self._read_header():
            return False

        def copy_arr(inp, out):
            for i in range(len(inp)):
                out[i] = inp[i]

        if self.lsize <= 9:
            self.datalen = self.lsize * self.flt3len
        else:
            mem2 = self._fxtc.read(self.hdr2)
            xdrfile = xdrlib.Unpacker(mem2)
            prec = xdrfile.unpack_float()
            self._frd.inv_p = 1. / prec
            int_arr = [0] * 8
            for i in range(len(int_arr)):
                int_arr[i] = xdrfile.unpack_int()
            nbytes = int_arr[-1]
            copy_arr(int_arr[:3], self._frd.minint)
            copy_arr(int_arr[3:6], self._frd.maxint)
            self._frd.smli = int_arr[6]
            self._frd.natoms = self.nacur
            self.datalen = self._align4(nbytes)
        if hdr_only:
            self._fxtc.seek(self.datalen, os.SEEK_CUR)
        else:
            self._decompess()
        if no_check:
            return True
        tcur = self.t
        if self.dt > 0:
            self.iframe += 1
            tnext = self._tnext()
            if self._bigdt:
                self._search_t(tnext)
            else:
                self._skip_to(tnext)
        self.t = tcur
        return True

    def _decompess(self):
        datalen = self.datalen
        nacur = self._frd.natoms
        X = self.X
        if X is None or X.shape[0] < nacur:
            X = self.X = np.zeros((nacur, 3), dtype=np.float32)
        if self.lsize <= 9:
            X[:] = np.fromfile(self._fxtc, dtype='>f', count=datalen // 4)
        else:
            count8 = (datalen + 7) & (~0x7)
            self._buf = np.empty((count8,), dtype=np.int8)
            self._buf[:datalen] = np.fromfile(self._fxtc, dtype=np.int8, count=datalen)
            pad_size = 8 - len(self._buf) % 8 if len(self._buf) % 8 != 0 else 0
            padded_int8_values = np.pad(self._buf, (0, pad_size), mode='constant', constant_values=0)
            reshaped_array = padded_int8_values.reshape(-1, 8)
            int64_array = reshaped_array.view(np.int64)
            ok, coords = unpack_frame(self._frd, int64_array, X)
            for coord in coords:
                print(coord)
            if not ok:
                raise XTCBadFrame("Bad frame data at time {:g}!".format(self.t))

    def eot(self):
        if self._fxtc.tell() >= self._fsize:
            return True
        if self.end > 0:
            tnext = self._tnext()
            if (tnext - self.end > self._teps):
                return True
        return False

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.X = None
        self._fxtc.close()

    def __iter__(self):
        eot = self.eot()
        while not eot:
            self.next_frame()
            yield self.X, self.box, self.t
            eot = self.eot()


def test_xtc_reader():
    xtc_reader = XTCReader("water-md.xtc")
    while 1:
        xtc_reader.next_frame()
        if xtc_reader.eot():
            break
    xtc_reader.__exit__(None, None, None)


def main():
    for i in range(0, 1):
        test_xtc_reader()


main()
