import numpy as np
from constructor import Constructor


'''Functions to deal with phase optimisation'''


def gen_U_from_param_array(test_params:list,
                            dim:int, 
                            pattern:list, 
                            final_phase: list or None
                            ) -> np.array:

    ''' 
    Generate unitary matrix from circuit

    test_params: list of test parameters
    dim: total system dimenstion
    pattern: pattern of unitcell elements
    final_phases: list of final phases not included in optimisation. If None, first dim elements of params correspond to these phases
    '''
    
    if not final_phase:
        final_phase = list(test_params[:dim])
    
    
    
    total_params = list(test_params[dim:])
    
    params = []

    for i,p in enumerate(total_params):

        if i%2:
            temp.append(p)
            params.append(temp)
        else:
            temp = []
            temp.append(p)
        

   
    system = Constructor(dim,pattern,params,final_phase)
    
    return system.total_U()


def cost_func(test_params:list,
              dim:int, 
              target_U:np.array, 
              pattern:list, 
              final_phase: list or None
              ) -> float:
    
    ''' 
    Cost function for phase optimisation

    test_params: list of test parameters
    dim: total system dimenstion
    target_U: desired unitary transformation
    pattern: pattern of unitcell elements
    final_phases: list of final phases not included in optimisation. If None, first dim elements of params correspond to these phases
    '''
    
    
    
    W = gen_U_from_param_array(test_params,dim,pattern,final_phase)
    U = target_U

    return np.sum(abs(W-U)**2)


def fidelity(M_opt:np.array, M: np.array) -> float:

    '''Calculates fidelity between two arbitrary matrices'''

    num = np.matrix.trace(M_opt.conj().T @ M)
    denom = np.sqrt(np.matrix.trace(M_opt.conj().T @ M_opt))*np.sqrt(np.matrix.trace(M.conj().T @ M))
    return np.real(num/denom)