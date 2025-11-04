import numpy as np
import scipy.sparse  as sp
from scipy.optimize import minimize
from tqdm import tqdm
import qutip as qp # doco: https://qutip.readthedocs.io/en/qutip-5.1.x/apidoc/apidoc.html


def build_lower_state_op(n = 2):
    """
    Function which builds and returns the annihilation operator without the coefficient.
    
    Parameters:
    - n: dimenstion of Hilbert space.

    Returns:
    lower_state: The lower state operator.
    """
    
    lower_state = np.ones(n - 1, dtype = complex)
    rows = np.arange(0, n - 1)
    cols = np.arange(1, n)
    lower_state = sp.coo_matrix((lower_state, (rows, cols)), shape=(n, n))
    lower_state = qp.Qobj(lower_state, dims=[[n], [n]])
    return lower_state

def bulid_rise_state_op(n = 2):
    """
    Function which builds and returns the creation operator without the coefficient. The highest possible state will be sent to himself.
    
    Parameters:
    - n: dimenstion of Hilbert space.

    Returns:
    rise_state: The rise state operator.
    """
    
    rise_state = np.ones(n - 1, dtype = complex)
    rows = np.arange(1, n)
    cols = np.arange(0, n - 1)
    rise_state = sp.coo_matrix((rise_state, (rows, cols)), shape=(n, n))
    rise_state = qp.Qobj(rise_state, dims=[[n], [n]])
    return rise_state

def bulid_SPRINT_op(n = 2):
    
    """
    Function which builds and returns the SPRINT operators.
    
    Parameters:
    - n: dimenstion of Hilbert space.

    Returns:
    Array of the SPRINT operators
    """
    
    ket_0 = qp.basis(2,0)
    ket_1 = qp.basis(2,1)
    r_r = ket_0 * ket_0.dag() 
    r_t = ket_1 * ket_0.dag()   # Projection operator from R state of the atom to T state
    t_t = ket_1 * ket_1.dag()   # Projection operator from T state of the atom to T state

    lower_state = build_lower_state_op(n) # Building the anhillation operator without the coefficient
    
    rise_state = qp.ket2dm(qp.basis(n,n - 1)) # Building the creation operator without the coefficient. Sends \ket{N} \to \ket{N}
    rise_state += bulid_rise_state_op(n)

    s_e = qp.tensor(r_r, qp.fock_dm(n,0), qp.qeye(n),qp.qeye(n),qp.qeye(n)) + qp.tensor(r_t, lower_state, rise_state,qp.qeye(n),qp.qeye(n)) # Building the operator acting on the early time bin given the atom state
    s_l = qp.tensor(r_r, qp.qeye(n),qp.qeye(n),qp.fock_dm(n,0), qp.qeye(n)) + qp.tensor(r_t, qp.qeye(n),qp.qeye(n),lower_state, rise_state) # Building the operator acting on the late time bin given the atom state
    
    deactived_atom = qp.tensor(t_t, qp.qeye(n),qp.qeye(n),qp.qeye(n),qp.qeye(n))                                                            # The operator representing the action of the atom in T state
    sprint_e = s_e + deactived_atom                                                                                                         # Building the full sprint operator for the early bin
    sprint_l = s_l + deactived_atom                                                                                                         # Building the full sprint operator for the late bin

    return [sprint_e,sprint_l]

def bulid_POVM(n = 2):
    
    """
    Function which builds and returns the POVM operators.
    
    Parameters:
    - n: dimenstion of Hilbert space.

    Returns:
    povm_ops: Array of the POVM operators
    """

    povm_ops = []
    rho_0 = qp.ket2dm(qp.basis(n,0))
    rho_other = qp.qeye(n) - rho_0
    povm_ops.append(qp.tensor(rho_0,rho_0))
    povm_ops.append(qp.tensor(rho_0,rho_other))
    povm_ops.append(qp.tensor(rho_other,rho_0))
    povm_ops.append(qp.tensor(rho_other,rho_other))
    return povm_ops

def measure_state(dens_mat,povm_ops):
    """
    Operates on the given density matrix using the given POVMs and builds a new state using the measuring results.

    Parameters:
    - dens_mat: The density matrix of the operator on which the POVMs act.
    - povm_ops: An array of the POVM operators

    Returns:
    The new density matrix
    """
    dens_mat = povm_ops[4] * dens_mat * povm_ops[4].dag()
    coeff_00 = np.sqrt((povm_ops[0] * dens_mat).tr())
    coeff_01 = np.sqrt((povm_ops[1] * dens_mat).tr())
    coeff_10 = np.sqrt((povm_ops[2] * dens_mat).tr())
    coeff_11 = np.sqrt((povm_ops[3] * dens_mat).tr())

    return (coeff_00 * qp.fock(4,0) + coeff_01 * qp.fock(4,1) + coeff_10 * qp.fock(4,2) + coeff_11 * qp.fock(4,3))

def build_post_selcet_op(n = 2,lowest_state = 0):
    """
    Function which builds and returns the post selection operator.
    
    Parameters:
    - n: dimenstion of Hilbert space.
    - lowest_state: The cut-off state of the post selection. If lowest_state >= n, sends the state to ket{0}.

    Returns:
    post_selection_op: The post selection operator
    """

    if(lowest_state == 0):
        post_selection_op = qp.tensor(qp.qeye(2),qp.qeye(n),qp.qeye(n),qp.qeye(n),qp.qeye(n))
        return post_selection_op

    elif(lowest_state >= n):
        post_selection_op = qp.qzero([2,n,n,n,n])
        post_full = post_selection_op.full()           # Convert to numpy array
        post_full[0,:] = 1            # Fill first column with ones
        post_selection_op = qp.Qobj(post_full)         # Convert back to Qobj
        return post_selection_op

    else:
        post_selection_op = qp.qzero([2,n,n,n,n])
        
        for n1 in range(n):
            for n2 in range(n):
                if n1 + n2 >= lowest_state:
                    post_selection_op += qp.tensor(qp.qeye(2),qp.fock_dm(n, n1),qp.qeye(n), qp.fock_dm(n, n2),qp.qeye(n))
        
        return post_selection_op

def bulid_beam_splitter(n = 2,theta = 0):
    
    """
    Builds and returns the beam splitter operator.

    Parameters:
    - theta: Float. The angle which defines the beam splitter. If theta = 0, the identity matrix is returned.
    - n: Hilbert space dimension.

    Returns:
    beam_splitter: Beam splitter operator on a tensor product space (n x n).
    """
    
    if(theta != 0):
        beam_splitter = qp.tensor(qp.create(n), qp.destroy(n)) + qp.tensor(qp.destroy(n), qp.create(n))
        beam_splitter = (1j * theta * beam_splitter).expm()
    else:
        beam_splitter = qp.tensor(qp.qeye(n),qp.qeye(n))

    return beam_splitter

def build_input_state(a,b,alpha,post_select_op,lowest_state = 0,n = 2):
    """
    Function which builds and returns the coherent input state.
    
    Parameters:
    - a: The value of the weight of the early time bin.
    - b: The value of the weight of the late time bin.
    - alpha: The value of alpha defining the coherent state. If alpha = 0, reutrn ket{0,0,0,0,0}.
    - lowest_state: The cut-off state of the post selection.
    - n: dimenstion of Hilbert space.

    Returns:
    post_selection_op: The post selection operator
    """
    if(alpha == 0):
        input_state = qp.tensor(qp.basis(2,0),qp.basis(n, 0),qp.basis(n,0), qp.basis(n, 0),qp.basis(n,0))
    else:
        input_state = (post_select_op * qp.tensor(qp.basis(2,0),qp.coherent(n, a * alpha),qp.basis(n,0), qp.coherent(n, b * alpha),qp.basis(n,0))).unit()
    return input_state

def shifted_coherent_state(n = 2, alpha = 0, lowest_state = 0):
    """
    Builds a coherent state such that the lowest state included is lowest_state.

    Parameters:
    - alpha (complex): alpha value for the coherent state. If alpha is set to zero, returns ket{0}. (0 by default)
    - lowest_state (int): Lowest state to be included in the coherent state. (0 by default)
    - n (int): The dimension of the Hilbert space. (2 by default)

    Returns:
        Qobj: The shifted and renormalized state.
    """

    if(alpha == 0 or lowest_state >= n):
        return qp.basis(n,0)

    full_state = qp.coherent(n, alpha, method = 'analytic')
    
    # Zero out components below `lowest_state`
    coeffs = full_state.full().flatten()
    coeffs[:lowest_state] = 0
    
    # Create new Qobj state
    return (qp.Qobj(coeffs, dims=[[n], [1]])).unit()

def sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints = 1,mid_reset = False, lowest_state = 0,n = 2):

    """
    A function which recieves values for a and alpha, and calculates the resulting fidelities. The expected state convention is ket{}_S ket{}_{T,e} ket{}_{R,e} ket{}_{T,l} ket{}_{R,l}

    Parameters:
    - input_state: The input state to the system.
    - sprint_ops: The SPRINT operators used in the sim
    - povm_ops: An array of the POVM operators used in the sim
    - beam_splitter: The BS operator used in the sim
    - num_of_sprints: The number of times the state will go through the SPRINT system. (1 by default)
    - mid_reset: Dictates wheter or not the SPRINT atom will be reset between the early and late time bins. (False by default)
    - lowest_state: Dictates the lowest possible state in the coherent states entering the system. (0 by default)
    - n: The size of the Hilbert space

    Returns:
    fid_T, fid_R - The values of the fidelities for the T and R arms
    """

    output_state = input_state
    R_projection = qp.tensor(qp.basis(2,0) * (qp.basis(2,0).dag() + qp.basis(2,1).dag()),qp.qeye(n),qp.qeye(n),qp.qeye(n),qp.qeye(n))
    for i in range(0,num_of_sprints):
        output_state = (R_projection * output_state).unit()
        output_state = (sprint_ops[0] * output_state).unit()
        if(mid_reset):
            output_state = (R_projection * output_state).unit()
        output_state = (sprint_ops[1] * output_state).unit()

    input_state = qp.ptrace(input_state, [1,3])
    input_state = measure_state(input_state,povm_ops)

    output_state = qp.ptrace(output_state,[1,2,3,4])
    
    output_state = beam_splitter * output_state * beam_splitter.dag()

    state_T,state_R = qp.ptrace(output_state,[0,2]), qp.ptrace(output_state,[1,3])
    state_T = measure_state(state_T,povm_ops)
    state_R = measure_state(state_R,povm_ops)

    fid_T = qp.fidelity(input_state,state_T)
    fid_R = qp.fidelity(input_state,state_R)

    return fid_T, fid_R

def get_fid_values(a_values,b_values = None,alpha_values = None,theta = 0,num_of_sprints = 1,mid_reset = False,lowest_state = 0,n = 2):
    
    """
    A function which recieves an array of a values or alpha values, and calculates the resulting fidelities.
    Either a_values or alpha_values should be an array, but not both.
    
    Parameters:
    - a_values: An array of the values of a.
    - b_values: An array of the values of b (if provided with an empty array, the function uses b = sqrt(1 - |a|^2)).
    - alpha_values: An array of the values of alpha.
    - sprint_ops: The SPRINT operators used in the sim.
    - povm_ops: An array of the POVM operators used in the sim.
    - beam_splitter: The BS operator used in the sim.
    - num_of_sprints: The number of times the state will go through the SPRINT system. (1 by default)
    - mid_reset: Dictates wheter or not the SPRINT atom will be reset between the early and late time bins. (False by default)
    - lowest_state: Dictates the lowest possible state in the coherent states entering the system. (0 by default)
    - n: The size of the Hilbert space

    Returns:
    fid_T, fid_R: The array of values of the fidelities for the T and R arms
    """

    if(isinstance(alpha_values, np.ndarray) and isinstance(a_values, np.ndarray)):
        raise Exception("Either a_values or alpha_values should be an array, not both.")

    if(not isinstance(b_values, np.ndarray) or b_values.size == 0):
        b_values = np.sqrt(1 - np.abs(a_values)**2)

    if(isinstance(b_values, np.ndarray) and isinstance(a_values, np.ndarray) and a_values.size != b_values.size):
        raise Exception("The sizes of a_values and b_values arrays should be the same.")

    sprint_ops = bulid_SPRINT_op(n)
    povm_ops = bulid_POVM(n)
    beam_splitter = bulid_beam_splitter(n = n,theta = theta) # Gets a beam splitter which operates on one time bin
    beam_splitter = qp.tensor(beam_splitter,beam_splitter) # Creates an operator which uses a beam splitter on each time bin
    post_select_op = build_post_selcet_op(n = n,lowest_state = lowest_state)

    if(isinstance(alpha_values, np.ndarray)):
        a = a_values
        b = b_values
        if(a == 0 or b == 0):
            povm_ops.append(bulid_beam_splitter(n = n,theta = 0))
        else:
            povm_ops.append(bulid_beam_splitter(n = n,theta = np.arctan(np.abs(b) / np.abs(a))))
        fid_T = np.empty_like(alpha_values,dtype = float)
        fid_R = np.empty_like(alpha_values,dtype = float)
        for i in tqdm(range(alpha_values.shape[0])):
            alpha = alpha_values[i]
            input_state = build_input_state(a,b,alpha,post_select_op,lowest_state,n)
            fid_T[i], fid_R[i] = sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints,mid_reset,lowest_state,n)
        return fid_T, fid_R

    elif(isinstance(a_values, np.ndarray)):
        alpha = alpha_values
        fid_T = np.empty_like(a_values,dtype = float)
        fid_R = np.empty_like(a_values,dtype = float)
        for i in tqdm(range(a_values.shape[0])):
            a = a_values[i]
            b = b_values[i]
            if(a == 0 or b == 0):
                povm_ops.append(bulid_beam_splitter(n = n,theta = 0))
            else:
                povm_ops.append(bulid_beam_splitter(n = n,theta = np.arctan(np.abs(b) / np.abs(a))))
            input_state = build_input_state(a,b,alpha,post_select_op,lowest_state,n)
            fid_T[i], fid_R[i] = sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints,mid_reset,lowest_state,n)
        return fid_T, fid_R

    else:
        a = a_values
        b = b_values
        alpha = alpha_values
        input_state = build_input_state(a,b,alpha,post_select_op,lowest_state,n)
        fid_T, fid_R = sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints,mid_reset,lowest_state,n)
        return fid_T, fid_R
