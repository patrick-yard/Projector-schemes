import numpy as np
import abc

''''Classes for optical components and to construct total photonic circuit'''

class BaseClass(abc.ABC):
    
   
    def __init__(self, dim:int, mode:int, params: float or list):

        '''Unit cell class that will be used with phase shifter and beam-splitter elements
        
        dim: total system dimension
        mode: which mode to is the component acting on indexed starting at 0
        params: parameters used to define component unitary'''

        self.dim = dim
        self.mode = mode
        self.params = params

    @abc.abstractmethod
    def U(self,params:float or list) -> np.array:
        
        '''Constructs component unitary matrix with parameters given by params'''

        pass
    
    def global_U(self) -> np.array:

        '''Embed component unitary in a total system sized matrix'''
        
        modes = [self.mode,self.mode + 1]

        indices = np.ix_(modes,modes)
        
        base = np.eye(self.dim,dtype = complex)

        base[indices] = self.U(self.params)

        return base

class BeamSplitter(BaseClass):

    def __init__(self, dim:int, mode:int, phase:float):

        '''Class defines a beamsplitter with a variable reflectivity
        dim: total system dimension
        mode: beamsplitter acts between modes mode and mode + 1
        phase: reflectivity given by sin(phase/2)**2'''
        
        super().__init__(dim,mode,phase)


    def U(self,p:float) -> np.array:
        
        r = np.sin(0.5*p)
        t = np.cos(0.5*p)
        
        return np.array([[r,t],[t,-r]])


class PhaseShifter(BaseClass):

    def __init__(self, dim:int, mode:int, phase:float):
        
        '''Class defines a phase shift
        dim: total system dimension
        mode: indicates which mode the phase shift emparted on
        phase: value of phase shift in rads'''
        
        super().__init__(dim,mode-1,phase)


    def U(self,p:float) -> np.array:

        return np.array([[np.exp(1j*p),0],[0,1]])


class UnitCell(BaseClass):

    def __init__(self,dim:int, mode:int, params:float or list):
       
        '''Class defines two parameter building block - phase shifter followed by beamsplitter
        dim: total system dimension
        mode: phase shift on mode +1 , beamsplitter between modes mode and mode + 1
        params: list [phase shift, reflectivity]'''

        super().__init__(dim,mode,params)

    def U(self,p:list) -> np.array:

        
        PS = PhaseShifter(self.dim,self.mode + 1,p[0])
        BS = BeamSplitter(self.dim,self.mode,p[1])

        return BS.U(p[1]) @ PS.U(p[0])

class Constructor():

    def __init__(self,dim:int,pattern:list,params:list,final_phases:list):

        '''Class to construct arbitrary mesh of UnitCell components
        dim: total system dimension
        pattern: list of modes for the each element of the circuit. Circuit built left to right
        params: list of tuples of the corresponding parameters for each unit cell in pattern
        final_phases: array of final phase shifts.'''

        self.dim = dim
        self.pattern = pattern
        self.params = params
        self.final_phases = final_phases


    def total_U(self) -> np.array:

        U_tot = np.eye(self.dim,dtype = complex)      
        
        
        if self.final_phases:
            
            phase_comps = [PhaseShifter(self.dim,i,p) for i,p in enumerate(self.final_phases)]

            for phase in phase_comps:
                
                U_tot = U_tot @ phase.global_U()
        
        
        comp_array = [UnitCell(self.dim,mode,p) for mode,p in zip(self.pattern,self.params)]
         
        
        comp_array.reverse()

           
        for i,comp in enumerate(comp_array):
            
          
            U_tot = U_tot @ comp.global_U()

        
        return U_tot


