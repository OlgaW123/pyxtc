# -*- coding: utf-8 -*-

# Copyright (c) 2020, Nikolay A. Krylov
# All rights reserved.

import os
import glob
import struct
import xdrlib
from ctypes import CDLL, c_bool, c_int, c_float, c_void_p, Structure, POINTER
import numpy as np

# frame_data C struct:
# struct frame_data {
#     int minint[3];
#     int maxint[3];
#     unsigned int smli;
#     const int natoms;
#     float inv_p;
#     int nt;
# };


class frame_data(Structure):
    _fields_ = [
                ("minint", 3 * c_int),
                ("maxint", 3 * c_int),
                ("smli", c_int),
                ("natoms", c_int),
                ("inv_p", c_float),
                ("nt", c_int),
    ]


__cdir__ = os.path.dirname(os.path.abspath(__file__))
l = glob.glob(os.path.join(__cdir__, "_xtc*"))

nxtc = len(l)

if nxtc < 1:
    raise RuntimeError("No helper library found!")

for lib in l:
    try:
        _xtc = CDLL(lib)
        break
    except Exception as e:
        pass

unpack_frame = _xtc.unpack_frame
unpack_frame.argtypes = [POINTER(frame_data), c_void_p, c_void_p]
unpack_frame.restype = c_bool


class XTCBadMagic(IOError):

    def __init__(self):
        super(XTCBadMagic, self).__init__("Bad frame magic!")


class XTCBadHeader(IOError):

    def __init__(self):
        super(XTCBadHeader, self).__init__("Bad frame header!")


class XTCBadFrame(IOError):
    pass


class XTCReader:
    """
    Main class to access MD trajectory data stored in xtc format.

    Supports context manager and generator protocols.
    Refer to the readme and tests for usage examples.

    Attributes
    ----------

    X : numpy.ndarray
        the atom coordinates of the current frame.
    box : numpy.ndarray
        the unit cell vectors of the current frame.
    t : float
        the time step of the current frame.
    """
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

    @staticmethod
    def _get_npa_ptr(npa):
        ptr = npa.__array_interface__['data'][0]
        return c_void_p(ptr)

    def __init__(self, fname, beg=-1, end=-1, dt=-1, nt=4):
        """
        Parameters
        ----------
        fname : str
            Input file name
        beg : float, optional
            The first frame to read from file (ps)
        end : float, optional
            The last frame to read from file (ps)
        dt : float, optional
            The interval between consecutive frames (ps)
        nt : int, optional
            Thr number of CPU to use
        """

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

        # Prepare - first, last ,dt, check dt
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
        # XXX: search backwards?
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
        """
        Read the next frame from trajectory file.

        Atom coordinates, box vectors and frame time is stored in X, box and t attributes respectively.
        Returns True on success and False on error.
        """

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
            ok = unpack_frame(self._frd, self._get_npa_ptr(self._buf), self._get_npa_ptr(X))
            if not ok:
                raise XTCBadFrame("Bad frame data at time {:g}!".format(self.t))

    def eot(self):
        """Returns True if end of trajectory is reached."""

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
        """
        Generator protocol support method.

        On each iteration yields tuple with X, box and t attributes.
        """
        eot = self.eot()
        while not eot:
            self.next_frame()
            yield self.X, self.box, self.t
            eot = self.eot()
