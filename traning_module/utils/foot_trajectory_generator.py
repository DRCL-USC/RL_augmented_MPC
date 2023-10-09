""" Foot Trajectory Generator Class """

import numpy as np

"""
    for (size_t j = 0; j < 4; j++) {
      double dh = 0.0;
      if (pi_[j] > 0.0) {
        double t = pi_[j] / M_PI_2;
        if (t < 1.0) {
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (-2 * t3 + 3 * t2);
        } else {
          t = t - 1;
          double t2 = t * t;
          double t3 = t2 * t;
          dh = (2 * t3 - 3 * t2 + 1.0);
        }
        dh *= clearance_[j];
      }

"""
class FootTrajectoryGenerator(object):
    """ Foot trajectory generator (z only), includes phase state. """
    def __init__(self,
                T=0.3, # foot trajectory period
                max_foot_height=0.05,
                dt=0.001,
                ):

        # maximum foot height
        self.h = max_foot_height 
        # nominal phase offsets
        self.phi_i0 =  np.array([np.pi, 0, 0, np.pi])#np.zeros(4)

        self.T = T
        # nominal frequency in Hz
        self.f0 = 1 / T #1.25
        # current phase (assume time at 0)
        self.phi_i = self.phi_i0 # np.zeros(4)
        # curret foot height deltas
        self.foot_dhs = np.zeros(4)
        # FTG dt
        self.dt = dt

    def setPhaseOffsets(self, phi_i0):
        """Set phase offsets. """
        self.phi_i0 = phi_i0

    def getPhases(self):
        return self.phi_i

    def getNominalFrequency(self):
        return self.f0
        
    def getNominalPeriod(self):
        return self.T

    def setMaxFootHeight(self, h):
        """ Set the maximum height of the swing. """
        self.h = h

    def getDeltaFootHeight(self, phase):
        """ Get the delta foot height according to the phase. 

        Phase in [0,2pi)
        """
        k = 2 * (phase - np.pi) / np.pi
        dh = -0.28 # delta height
        # if phase > 0:
        if k > 0: # means phase > pi, so in swing
            #k = phase / (2*np.pi) # old, from ETH
            if k < 1: # swing up
                dh += self.h * (-2*k**3 + 3*k**2 )
            elif k > 1 and k < 2 : # swing down
                dh += self.h * ( 2*k**3 - 9*k**2 + 12*k - 4 )
            else:
                dh = -0.28
        return dh

    def setPhasesAndGetDeltaHeights(self, t, fi=None, fi_period=None):
        """ Set phases phi_i according to current time, and desired frequency 

        phi = (phi_i0 + (f0+fi)*t) mod (2*pi)
        """
        if fi_period is not None:
            fi = 1/fi_period
        # self.phi_i = self.phi_i0 + ( (2*np.pi*self.f0) + (2*np.pi*fi) ) * t
        self.phi_i += ( (2*np.pi*self.f0) + (2*np.pi*fi) ) * self.dt # t
        self.phi_i = self.phi_i % (2*np.pi)
        # print(self.phi_i)
        #calculate foot heihts
        return self.calcFootHeights()

    def calcFootHeights(self):
        """From current phases, get foot heights """
        self.foot_dhs = np.zeros(4)
        for i in range(4):
            self.foot_dhs[i] = self.getDeltaFootHeight(self.phi_i[i])
        return self.foot_dhs
