
import numpy as np
from constructor import Constructor


'''Functions to deal with phase optimisation'''


def gen_ideal_U_from_param_array(test_params:list,
                            dim:int, 
                            pattern:list, 
                            out_phase: bool,
                            in_phase: bool,
                            zero_phases: list or None
                            ) -> np.array:

    ''' 
    Generate unitary matrix from circuit

    test_params: list of test parameters
    dim: total system dimenstion
    pattern: pattern of unitcell elements
    end_phases: bool whether end phases are included in test_params. if included first dim elems of test params
    in phase: bool whether input phases are included in test_params. if included first dim elems of test params 
    if both input and output phases included test_params  = end_phases + in_phases + params
    zero_phases: list of bools indicating which external phases should be 0 if None all external phases are present.
    '''
    
    if out_phase and in_phase:

        out_phase = list(test_params[:dim])
        in_phase = list(test_params[dim:2*dim])
        total_params = list(test_params[2*dim:])
    
    elif in_phase:

        in_phase = list(test_params[:dim])
        total_params = list(test_params[dim:])
    
    elif out_phase:

        out_phase = list(test_params[:dim])
        total_params = list(test_params[dim:])

    else:

        total_params = test_params
    
    
    params = []

    for i,p in enumerate(total_params):

        if i%2:
            temp.append(p)
            params.append(temp)
        else:
            temp = []
            temp.append(p)
        

   
    system = Constructor(dim,pattern,params,out_phase,in_phase)
    
    return system.ideal_total_U(zero_phases=zero_phases)



def cost_func(test_params:list,
              dim:int, 
              target_U:np.array, 
              pattern:list, 
              out_phase: bool,
              in_phase: bool,
              zero_phases = None
              ) -> float:
    
    ''' 
    Cost function for phase optimisation

    test_params: list of test parameters
    dim: total system dimenstion
    target_U: desired unitary transformation
    pattern: pattern of unitcell elements
    end_phases: bool whether end phases are included in test_params.
    in phase: bool whether input phases are included in test_params.
    zero_phases: list of bools indicating which external phases should be 0 if None all external phases are present.

    '''
    
    W = gen_ideal_U_from_param_array(test_params,dim,pattern,out_phase,in_phase,zero_phases)
    U = target_U

    return np.sum(abs(W-U)**2)

def subset_cost_func(test_params:list,
              dim:int, 
              target_U:np.array, 
              pattern:list, 
              out_phase: bool,
              in_phase: bool,
              sub_indices:np.array,
              zero_phases: list or None
              ) -> float:

    '''cost function for only selected rows or columns
    
    test_params: list of test parameters
    dim: total system dimenstion
    target_U: desired unitary transformation
    pattern: pattern of unitcell elements
    end_phases: bool whether end phases are included in test_params.
    in phase: bool whether input phases are included in test_params.
   
    final_phases: list of final phases not included in optimisation. If None, first dim elements of params correspond to these phases
    sub_indices: indices to select subset of total unitary to be compared
    zero_phases: list of bools indicating which external phases should be 0 if None all external phases are present.
    
    '''
    W = np.eye(dim,dtype = complex)
    U = np.eye(dim,dtype = complex)
    
    W_tot = gen_ideal_U_from_param_array(test_params,dim,pattern,out_phase,in_phase,zero_phases)

    W[sub_indices] = W_tot[sub_indices]
    # print(W_tot[sub_indices].shape)
    # print()
    if W_tot[sub_indices].shape != target_U.shape:
        U[sub_indices] = target_U[sub_indices]
    else:
        U[sub_indices] = target_U

    return np.sum(abs(W-U)**2)



def fidelity(M_opt:np.array, M: np.array) -> float:

    '''Calculates fidelity between two arbitrary matrices'''
 
    
    num = np.matrix.trace(M_opt.conj().T @ M)
    denom = np.sqrt(np.matrix.trace(M_opt.conj().T @ M_opt))*np.sqrt(np.matrix.trace(M.conj().T @ M))

    return np.real(num/denom)



def fidelity_sub(M_opt:np.array, M: np.array,sub_indices:np.array, dim:int) -> float:

    '''Calculates fidelity between two arbitrary matrices'''
 
    W = np.eye(dim,dtype = complex)
    U = np.eye(dim,dtype = complex)

    W[sub_indices] = M_opt[sub_indices]
    
    if M_opt[sub_indices].shape != M.shape:
        U[sub_indices] = M[sub_indices]
    else:
        U[sub_indices] = M
   
   
    num = np.matrix.trace(W.conj().T @ U)
    denom = np.sqrt(np.matrix.trace(W.conj().T @ W))*np.sqrt(np.matrix.trace(U.conj().T @ U))

    return np.real(num/denom)