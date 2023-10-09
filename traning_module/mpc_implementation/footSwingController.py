import numpy as np
from mpc_implementation.bezier_util import cubicBezier, cubicBezierFirstDerivative

class footSwingController(object):
    def __init__(self):
        self._height = 0.05
        self._p0 = np.zeros((1,3), dtype=np.float32)  # initial pos
        self._pf = np.zeros((1,3), dtype=np.float32)  # final pos
        self._p = np.zeros((1,3), dtype=np.float32)   # foot position to be returned
        self._v = np.zeros((1,3), dtype=np.float32)
    
    def set_swing_height(self, h):
        self._height = h

    def set_init_position(self, p0):
        self._p0 = p0

    def set_final_position(self, pf):
        self._pf = pf

    def computeSwingTrajectory(self, phase, swingtime):
        self._p = cubicBezier(self._p0, self._pf, phase)
        self._v = cubicBezierFirstDerivative(self._p0, self._pf, phase)/swingtime

        if phase < 0.5:
            zp = cubicBezier(self._p0[2], self._p0[2] + self._height, phase * 2)
            zv = cubicBezierFirstDerivative(self._p0[2], self._p0[2] + self._height, phase * 2)/(swingtime * 0.5)
        else:       
            zp = cubicBezier(self._p0[2] + self._height, self._pf[2], phase * 2 -1)
            zv = cubicBezierFirstDerivative(self._p0[2] + self._height, self._pf[2], phase * 2 -1)/(swingtime * 0.5)

        self._p[2] = zp
        self._v[2] = zv
