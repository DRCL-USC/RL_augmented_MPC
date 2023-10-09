""" Interpolation.h translation  """

import numpy as np

def lerp(y0, yf, x0):
    """ Linear interpolation betweeo y0 and yf. x is between 0 and 1 """
    assert(x >=0 and x<=1)
    return y0 + (yf-y0) * x

def cubicBezier(y0, yf, x):
    """ Cubic bezier interpolation between y0 and yf.  x is between 0 and 1 """
    assert(x >= 0 and x <= 1);
    yDiff = yf - y0
    bezier = x * x * x + 3 * (x * x * (1 - x))
    return y0 + bezier * yDiff

def cubicBezierFirstDerivative(y0, yf, x):
    """ Cubic bezier interpolation derivative between y0 and yf.  x is between 0 and 1 """
    assert(x >= 0 and x <= 1)
    yDiff = yf - y0;
    bezier = 6*x * (1 - x)
    return bezier * yDiff

def cubicBezierSecondDerivative(y0, yf, x):
    """ Cubic bezier interpolation derivative between y0 and yf.  x is between 0 and 1 """
    assert(x >= 0 and x <= 1);
    yDiff = yf - y0;
    bezier = 6 - 12*x;
    return bezier * yDiff


class FootSwingTrajectory(object):
    """ A foot swing trajectory for a single foot. """
    def __init__():
        self._p0 = np.zeros(3)
        self._pf = np.zeros(3)
        self._p  = np.zeros(3)
        self._v  = np.zeros(3)
        self._a  = np.zeros(3)
        self._height = 0

    def setInitialPosition(self, p0):
        """ Set the starting point of the foot. """
        self._p0 = p0

    def setFinalPosition(self, pf):
        """ Set the desired final position of the foot """
        self._pf = pf

    def setHeight(self, h):
        """ Set the maximum height of the swing. """
        self._height = h

    def getPosition(self):
        """ Get the foot position at the current point along the swing. """
        return self._p

    def getVelocity(self):
        """ Get the foot velocity at the current point along the swing. """
        return self._v

    def getAcceleration(self):
        """ Get the foot accleration at the current point along the swing. """
        return self._a

    def computeSwingTrajectoryBezier(self, phase, swingTime):
        """ Compute foot swing trajectory with a bezier curve. 
        Params: 
            phase: how far along we are in the swing (0 to 1)
            swingTime: how long the swing should take (seconds) 

        """
        self._p = cubicBezier(self._p0, self._pf, phase)
        self._v = cubicBezierFirstDerivative( self._p0, self._pf, phase) / swingTime
        self._a = cubicBezierSecondDerivative(self._p0, self._pf, phase) / (swingTime**2)

        if phase < 0.5:
            zp = cubicBezier(self._p0[2], self._p0[2] + self._height, phase*2)
            zv = cubicBezierFirstDerivative( self._p0[2], self._p0[2] + self._height, phase*2) * 2 / swingTime
            za = cubicBezierSecondDerivative(self._p0[2], self._p0[2] + self._height, phase*2) * 4 / (swingTime**2)
        else:
            zp = cubicBezier(self._p0[2] + self._height, self._pf[2], phase*2 - 1)
            zv = cubicBezierFirstDerivative( self._p0[2] + self._height, self._pf[2], phase*2 - 1) * 2 / swingTime
            za = cubicBezierSecondDerivative(self._p0[2] + self._height, self._pf[2], phase*2 - 1) * 4 / (swingTime**2)

        self._p[2] = zp
        self._v[2] = zv
        self._a[2] = za


"""
class Gait
{
public:
  Gait(int nMPC_segments, Vec4<int> offsets, Vec4<int>  durations, const std::string& name="");
  ~Gait();
  Vec4<double> getContactSubPhase();
  Vec4<double> getSwingSubPhase();
  int* mpc_gait();
  void setIterations(int iterationsPerMPC, int currentIteration);
  int _stance;
  int _swing;


private:
  int _nMPC_segments;
  int* _mpc_table;
  Array4i _offsets; // offset in mpc segments
  Array4i _durations; // duration of step in mpc segments
  Array4d _offsetsPhase; // offsets in phase (0 to 1)
  Array4d _durationsPhase; // durations in phase (0 to 1)
  int _iteration;
  int _nIterations;
  double _phase;

};

horizonLength = 10 (nMPC_segments)
0.001, 30
iterationsBetweenMPC = 30
dt = 0.001

  galloping(horizonLength, Vec4<int>(0,2,7,9),Vec4<int>(5,5,5,5),"Galloping"),
  pronking(horizonLength, Vec4<int>(0,0,0,0),Vec4<int>(4,4,4,4),"Pronking"),
  trotting(horizonLength, Vec4<int>(0,5,5,0), Vec4<int>(5,5,5,5),"Trotting"),
  bounding(horizonLength, Vec4<int>(5,5,0,0),Vec4<int>(5,5,5,5),"Bounding"),
  walking(horizonLength, Vec4<int>(0,3,5,8), Vec4<int>(5,5,5,5), "Walking"),
  pacing(horizonLength, Vec4<int>(5,0,5,0),Vec4<int>(5,5,5,5),"Pacing")
"""

class Gait:
    def __init__(self, 
                nMPC_segments, 
                offsets,        # offset in mpc segments
                durations,      # duration of step in mpc segments
                name ):
        self._offsets = offsets
        self._durations = durations
        self._nIterations = nMPC_segments
        self._mpc_table = np.zeros(nMPC_segments * 4)

        # offsets in phase (0 to 1)
        self._offsetsPhase = offsets / nMPC_segments
        # durations in phase (0 to 1)
        self._durationsPhase = durations / nMPC_segments

        self._stance = durations[0]
        self._swing = nMPC_segments - durations[0]
        self._phase = 0
        self._name = name

    def getContactsSubPhase(self):
        progress = self._phase - self._offsetsPhase
        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1
            if progress[i] > self._durationsPhase[i]:
                progress[i] = 0
            else:
                progress[i] = progress[i] / self._durationsPhase[i]
        return progress

    def getSwingSubPhase(self):
        swing_offset = self._offsetsPhase + self._durationsPhase
        for i in range(4):
            if swing_offset[i] > 1:
                swing_offset -= 1

        swing_duration = 1 - self._durationsPhase
        progress = self._phase - swing_offset

        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1
            if progress[i] > swing_duration[i]:
                progress[i] = 0
            else:
                progress[i] = progress[i] / swing_duration[i]

        return progress


    def mpc_gait(self):
        """For completeness and convention, not used """
        for i in range(self._nIterations):
            iter_i = (i + self._iteration + 1) % self._nIterations
            progress = iter_i - self._offsets
            for j in range(4):
                if progress[j] < 0:
                    progress[j] += self._nIterations
                if progress[j] < self._durations[j]:
                    self._mpc_table[i*4+j] = 1
                else:
                    self._mpc_table[i*4+j] = 0
        return self._mpc_table

    def setIterations(self,iterationsPerMPC, currentIteration):
        """In MPC, iterationsBetweenMPC=30 

        so as an example:
        _iteration = (curr_iteration / 30) % 10
        _phase = (curreIteration % (30*10)) / (30*10)
        so entire cycle is 300ms, 0r 0.3 s
        """
        self._iteration = (currentIteration / iterationsPerMPC) % self._nIterations
        self._phase = (currentIteration % (iterationsPerMPC * self._nIterations)) / (iterationsPerMPC * self._nIterations)