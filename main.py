from vis import *
from func import *


def main():
    n = 5                       # Determines the number of qubits used in the generation of the coherent states
    lowest_state = 0            # Determines the lowest number state that is included in the coherent states. Defualt is zero.
    theta = 0                   # The angle characterizing the beam splitter. Default is zero.
    mid_reset = False           # Determines whether or not we reset the atom between the early bin and late bin. Defualt is False.
    num_of_sprints = 1          # Determines the number of times the whole state (both early and late)  will go through the SPRINT system. Defualt is one.
    
    a = 1
    b = 0
    alpha_values = np.linspace(0, np.sqrt(1), 50,dtype = float)
    
    fid_T,fid_R = get_fid_values(a,b,alpha_values,theta,num_of_sprints,mid_reset,lowest_state,n)
    plot_fidelity(a,b,alpha_values,fid_T,fid_R,'square')


if __name__ == '__main__':
    main()
