import numpy as np

class gait(object):
    def __init__(self,
                 MPC_segments:int,   # MPC segments
                 offsets:np.ndarray,   # offsets for stance in  MPC segments (for each leg)
                 stance_durations:np.ndarray,  #duration of stance phase
                 gait_name:str):
        self._mpc_table = np.zeros(MPC_segments * 4)
        self._nIterations = MPC_segments

        self._offsets = offsets
        self._durations = stance_durations
        self._stance_duration = stance_durations[0]
        self._swing_duration = MPC_segments - self._stance_duration
        self._PhaseOffsets = offsets / MPC_segments  # offset in phase for each leg
        self._Stance_Duration_phase = stance_durations / MPC_segments  # stance duration in phase for each leg
        self._gaitName = gait_name
        self._iteration = 0 # step inside one cycle
        self._phase = 0 # current gait phase

    def setIterations(self, iterationsPerMPC, currentIteration):
        self._iteration = np.floor(currentIteration / iterationsPerMPC) % self._nIterations
        self._phase = currentIteration % (iterationsPerMPC * self._nIterations) / (iterationsPerMPC * self._nIterations)

    def getLegPhases(self): 
        current_gait_phase = np.array([self._phase] * 4)
        temp =  self._PhaseOffsets + current_gait_phase
        for i in range(4):
            if temp[i] > 1.0:
                temp[i] -= 1.0
        # temp = np.where(temp > 1.0, temp - 1.0, temp)
        return temp

    def getContactSubPhase(self):
        progress = np.array([self._phase] * 4) - self._PhaseOffsets
 
        for i in range(4):
            if progress[i] < 0:
                progress[i] += 1.0
            if progress[i] > self._Stance_Duration_phase[i]:
                progress[i] = 0
            else:
                progress[i] = progress[i] / self._Stance_Duration_phase[i]
            
        return progress

    def getSwingSubPhase(self):
        swing_offset = self._PhaseOffsets + self._Stance_Duration_phase
        # for i in range(4):
        #     if swing_offset[i] > 1:
        #         swing_offset[i] -= 1.0
        swing_offset = np.where(swing_offset > 1, swing_offset-1.0, swing_offset)
        
        swing_duration = np.array([1.]*4) - self._Stance_Duration_phase
        progress = np.array([self._phase]*4) - swing_offset

        for i in range(4):
            if progress[i] < 0: 
                progress[i] += 1.0
            if progress[i] > swing_duration[i]:
                progress[i] = 0
            
        # progress = np.where(progress < 0, progress+1.0, progress)
        # progress = np.where(progress > swing_duration, 0, progress)
            progress[i] = progress[i] / swing_duration[i]

        return progress

    def getMPCtable(self):
        for i in range(self._nIterations):
            iter = (i + self._iteration) % self._nIterations
            progress = np.array([iter]*4) - self._offsets
            for j in range(4):
                if progress[j] < 0:
                    progress[j] += self._nIterations
                if progress[j] < self._durations[j]:
                    self._mpc_table[i*4 + j] = 1
                else:
                    self._mpc_table[i*4 + j] = 0
            # progress = np.where(progress < 0, progress + self._nIterations, progress)
            # self._mpc_table = np.where(progress < self._durations, 1, 0)
            
        return self._mpc_table