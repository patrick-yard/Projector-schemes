import numpy as np

'''File containing functions to generate with qudit projectors'''

def get_gellmann_matrices(dim:int) ->list:

    '''Function to return all Gellmann matrices for a given dimension
        dim: dimension
        returns:
        full_list: combined list of all Gellmann matrices 
        '''

    comp_basis=np.identity(dim)

    symmetric_temp=[[np.outer(comp_basis[i], comp_basis[k]) +  np.outer(comp_basis[k], comp_basis[i])  for i in range(k)] for k in range(dim)]
    
    symmetric=[]
    
    for i in symmetric_temp:
        if len(i)>0:
            symmetric=symmetric+i

    antisymmetric_temp=[[-1.j*np.outer(comp_basis[i], comp_basis[k]) +  1.j*np.outer(comp_basis[k], comp_basis[i])  for i in range(k)] for k in range(dim)]
    antisymmetric=[]
    for i in antisymmetric_temp:
        if len(i)>0:
            antisymmetric=antisymmetric+i

    diagonal=[np.sqrt(2/((k+1)*(k+2)))* (sum( [np.outer(comp_basis[i], comp_basis[i]) for i in range(k+1)])- (k+1)*np.outer(comp_basis[k+1], comp_basis[k+1]) ) for k in range(dim-1)]
    
    full_list = [comp_basis]+diagonal+symmetric+antisymmetric
    
    return full_list


def get_bases_from_matrices(matr_list:list) -> np.array:

    '''Function returns eigenbases for a given list of matrices '''
    
    return  np.array([np.linalg.eig(basis)[1].T for basis in matr_list])

def get_eigenvalues_from_matrices(matr_list:list) -> np.array:

    '''Function returns eigenvalues for a given list of matrices'''

    return np.array([np.linalg.eig(basis)[0].T for basis in matr_list])


def construct_projector(eigen:np.array) -> np.array:

    '''Function to generate projector onto given eigenstates
        
        inputs

        eigen: basis eigenstates to be projected.

        return

        proj: projector'''
    
    # if len(eigen.shape) == 1:
    #     dim = len(eigen)
    #     eigen = [eigen]
    # else:
    dim = len(eigen[0])
    
    proj = np.eye(dim,dtype = complex)
    
    out_idx = [i for i in range(dim)]

    for i,e in enumerate(eigen):
        
        idx = np.ix_([i],out_idx)
        
        proj[idx] = e


    return proj

