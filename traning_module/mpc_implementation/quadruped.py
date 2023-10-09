import numpy as np
DTYPE = np.float16

class Quadruped:
    def __init__(self):
        self._abadLinkLength = 0.0838
        self._hipLinkLength = 0.2
        self._kneeLinkLength = 0.2
        # self._hipLinkLength = 0.25
        # self._kneeLinkLength = 0.25
        self._kneeLinkY_offset = 0.0
        self._bodyMass = 12.75

        self._bodyInertia = np.array([0.132, 0, 0, 
                                      0, 0.3475, 0, 
                                      0, 0, 0.3775], dtype=DTYPE)
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            # self._mpc_weights = [1., 1., 0, 0, 0, 20, 0., 0., .1, .1, .1, .0, 0]
        # self._mpc_weights = [0.5, 0.5, 10, 1.5, 1.5, 20, 0.0, 0.0, 0.2, 0.1, 0.1, 0.1, 0]
        self._mpc_weights = [0.25, 0.25, 10, 1.5, 1.5, 20, 0.0, 0.0, 0.2, 0.1, 0.1, 0.1, 0]