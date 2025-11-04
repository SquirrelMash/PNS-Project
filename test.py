from vis import *
from func import *

def test_shifted_coherent():
    n = 5    
    a = 1
    alpha = 1

    print(qp.coherent(n, a * alpha,method='analytic'))
    print(shifted_coherent_state(n, 0 , 2))

def test_sprint_sim():
    n = 5
    lowest_state = 0
    theta = 0
    mid_reset = False
    num_of_sprints = 1
    
    a = 1
    b = np.sqrt(1 - np.abs(a)**2)
    alpha = 1

    sprint_ops = bulid_SPRINT_op(n)
    povm_ops = bulid_POVM(n)
    if(a == 0 or b == 0):
        povm_ops.append(bulid_beam_splitter(n = n,theta = 0))
    else:
        povm_ops.append(bulid_beam_splitter(n = n,theta = np.arctan(np.abs(b) / np.abs(a))))
    beam_splitter = bulid_beam_splitter(n = n,theta=theta)
    beam_splitter = qp.tensor(beam_splitter,beam_splitter)
    input_state = 1/np.sqrt(2) * (qp.tensor(qp.basis(2,0),qp.basis(n, 1),qp.basis(n,0), qp.basis(n, 0),qp.basis(n,0)) - qp.tensor(qp.basis(2,0),qp.basis(n, 0),qp.basis(n,0), qp.basis(n, 1),qp.basis(n,0)))
    print(sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints,mid_reset,lowest_state,n))

def test_num_of_sprints():
    n = 5
    lowest_state = 0
    theta = 0
    mid_reset = False
    num_of_sprints = 0
    
    a = 1
    b = np.sqrt(1 - np.abs(a)**2)
    alpha = 1

    sprint_ops = bulid_SPRINT_op(n)
    povm_ops = bulid_POVM(n)
    if(a == 0 or b == 0):
        povm_ops.append(bulid_beam_splitter(n = n,theta = 0))
    else:
        povm_ops.append(bulid_beam_splitter(n = n,theta = np.arctan(np.abs(b) / np.abs(a))))
    beam_splitter = bulid_beam_splitter(n = n,theta=theta)
    beam_splitter = qp.tensor(beam_splitter,beam_splitter)
    input_state = qp.tensor(qp.basis(2,0),shifted_coherent_state(n, a * alpha,lowest_state),qp.basis(n,0), shifted_coherent_state(n, b * alpha,lowest_state),qp.basis(n,0))
    print(sim_sprint(input_state,sprint_ops,povm_ops,beam_splitter,num_of_sprints,mid_reset,lowest_state,n))


def test_post_select():
    n = 5
    lowest_state = 1
    a = 1
    b = np.sqrt(1 - np.abs(a)**2)
    alpha = 1

    post_select_op = build_post_selcet_op(n=n,lowest_state=lowest_state)

    print(qp.tensor(qp.basis(2,0),qp.fock(n,1),qp.basis(n,0),qp.fock(n,0),qp.basis(n,0)).dag() * (post_select_op * qp.tensor(qp.basis(2,0),qp.coherent(n, a * alpha),qp.basis(n,0), qp.coherent(n, b * alpha),qp.basis(n,0))).unit())

def test_main():
    #test_shifted_coherent()
    #test_sprint_sim()
    #test_num_of_sprints()
    test_post_select()



if __name__ == '__main__':
    test_main()