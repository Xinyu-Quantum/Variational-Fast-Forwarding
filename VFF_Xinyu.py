# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 15:48:53 2021

@author: xinyu
"""
import numpy as np
from math import pi
import pickle
from matplotlib import pyplot as plt
from scipy.integrate import simps 
from scipy import optimize as op
import time

from qiskit import*
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit import IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.opflow import I,X,Y,Z
from qiskit.extensions import HamiltonianGate

from mitiq import zne
from mitiq.zne import execute_with_zne
from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory, ExpFactory, PolyExpFactory, AdaExpFactory


IBMQ.save_account("7f647333d63485cd69118266daf5fabd1eb01ccc82ddca48ab094afbe21669a69db036d6dd5bde3e4f42ef6d2aa343280e87ca9389bdaed6c587ac1f9bfffd1b")
provider = IBMQ.load_account()

# define simulator backend: ideal simulator and statevector simulator
qasm_simulator = Aer.get_backend('qasm_simulator')
statevector_simulator = Aer.get_backend('statevector_simulator')


## LHST test circuit for single spin model
qubit_num = 1

#definition of the LHST test circuit
trotter_register = QuantumRegister(qubit_num, name = 'tro') #RegisterA
diag_register = QuantumRegister(qubit_num, name = 'diag' ) #RegisterB
classical_register = ClassicalRegister(2, name="c") #Classical register with 2 bits
circ = QuantumCircuit(trotter_register, diag_register, classical_register) #construct the LHST circuit

#generate maximal entangle state
def MES_generate(circ, qubit_num):
    for i in range(qubit_num):
        circ.h(i)
        circ.cx(i,i+qubit_num)
    circ.barrier()
    return 0

#inverse of MES_generate
def MES_reverse_generate(circ, qubit_num):
    for i in range(qubit_num):
        circ.cx(i,i+qubit_num)
        circ.h(i)
    circ.barrier()
    return 0
    
MES_generate(circ, qubit_num)

#Set the parameters of the Hamiltonian H = ax*sigmax+ay*sigmay+az*sigmaz
ax = 1.
ay = 1.
az = 1.
deltat = 0.1 #Delta t, a small time interval
a = [ax, ay, az]
a = np.array(a)
a = a/np.sqrt(np.sum(a*a))
optim_step = 500 #the numbe of optimization step
repeat_num = 1 #the times that an optimization is repeated
color = ['blue','red','green','yellow','pink','orange','cyan','violet']

#Trotterized Unnitary, act it on RegisterA
circ.rz(2*a[0]*deltat, 0)
circ.ry(2*a[1]*deltat, 0)
circ.rx(2*a[2]*deltat, 0)

#Diagonal factorization ansatz
#D:the digonal matrix
D_parameters = [Parameter('-γz')]
D_circ = QuantumCircuit(QuantumRegister(qubit_num), name = 'D*')
D_circ.barrier()
D_circ.rz(D_parameters[0], 0)
D_circ.barrier()

#W: unitary that encodes eigenvector
W_parameters = [Parameter('-θz'),Parameter('-θx')]
Wdag_parameters = [Parameter('θz'),Parameter('θx')]

W_circ = QuantumCircuit(QuantumRegister(qubit_num), name = 'W*')
W_circ.barrier()
W_circ.rz(W_parameters[0],0)
W_circ.rx(W_parameters[1],0)
W_circ.barrier()

#W^{\dag}, inverse of unitary W
Wdag_circ = QuantumCircuit(QuantumRegister(qubit_num), name = 'Wdag*')
Wdag_circ.barrier()
Wdag_circ.rx(Wdag_parameters[1],0)
Wdag_circ.rz(Wdag_parameters[0],0)
Wdag_circ.barrier()

#act ansatz on Register B
circ.append(W_circ.to_instruction(),[diag_register[0]])
circ.append(D_circ.to_instruction(),[diag_register[0]])
circ.append(Wdag_circ.to_instruction(),[diag_register[0]])
circ.barrier()

#circ.draw('mpl')

#set initial parameters
alpha_D = np.array([0.9]) #value of D_parameters
alpha_W = np.array([1.0,1.5]) #value of W_parameters

#set the shot number of measurement
shot_num = 100000

def Bind_parameter_circuit(alpha_D, alpha_W, alpha_Wdag):
    ##Bind parameters into the parameterized circuit to obtain an excutable circuit
    circuit = circ.bind_parameters({D_parameters[i]: alpha_D[i] for i in range(len(alpha_D))})
    circuit = circuit.bind_parameters({W_parameters[i]: alpha_W[i] for i in range(len(alpha_W))})
    circuit = circuit.bind_parameters({Wdag_parameters[i]: -alpha_Wdag[i] for i in range(len(alpha_Wdag))})
    return circuit

def LHST(alpha_D, alpha_W, alpha_Wdag, shot_num, qubit_num, type_sim, noise_model):
    ##calculate the LHST cost function
    #type_sim determines the type of simulator that is used
    circuit_base = Bind_parameter_circuit(alpha_D, alpha_W, alpha_Wdag)
    if(type_sim == 0):
            #state vector simulator
            circuit = circuit_base.copy()
            MES_reverse_generate(circuit, qubit_num)
            job = execute(circuit,backend = statevector_simulator)
            result = job.result().get_statevector()
            value = np.array([abs(result[0])**2*shot_num])
    else:
        value = np.zeros(qubit_num, dtype = float)
        for i in range(qubit_num):
            if(type_sim == 1):
                #noiseless quantum simulator, with no noise model
                circuit = circuit_base.copy()
                circuit.cx(i,i+qubit_num)
                circuit.h(i)
                circuit.barrier()
                circuit.measure([i,i+qubit_num],[0,1])
                job = execute(circuit,backend = qasm_simulator,shots=shot_num)     
                result = job.result()
                counts = result.get_counts(circuit)
                value[i] = counts.get('00')
            if(type_sim == 2):
                #noisy quantum simulator, with noise model applied
                circuit = circuit_base.copy()
                circuit.cx(i,i+qubit_num)
                circuit.h(i)
                circuit.barrier()
                circuit.measure([i,i+qubit_num],[0,1])                
                job = execute(circuit,backend = qasm_simulator,shots=shot_num,
                              basis_gates=noise_model.basis_gates,
                              noise_model=noise_model)
                result = job.result()
                counts = result.get_counts(circuit)
                value[i] = counts.get('00')
    value = value/shot_num
    return 1-np.mean(value)

def LHST_gradient_W(alpha_D, alpha_W, shot_num, qubit_num, backend_sim, noise_model):
    ##calculate the gradient of cost function with respect to W_parameters
    gradient_W = np.zeros(len(alpha_W),dtype = float)
    for i in range(len(alpha_W)):
        alpha1 = alpha_W.copy() 
        alpha1[i] += pi/2
        alpha2 = alpha_W.copy()
        alpha2[i] -= pi/2
        #parameter shift rule
        gradient_W[i] = 0.5*(LHST(alpha_D, alpha1, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
                             -LHST(alpha_D, alpha2, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
                             +LHST(alpha_D, alpha_W, alpha1, shot_num, qubit_num, backend_sim, noise_model)
                             -LHST(alpha_D, alpha_W, alpha2, shot_num, qubit_num, backend_sim, noise_model))
    return gradient_W

def LHST_gradient_D(alpha_D, alpha_W, shot_num, qubit_num, backend_sim, noise_model):
    ##calculate the gradient of cost function with respect to D_parameters
    gradient_D = np.zeros(len(alpha_D),dtype = float)
    for i in range(len(alpha_D)):    
        alpha1 = alpha_D.copy()
        alpha1[i] += pi/2
        alpha2 = alpha_D.copy()
        alpha2[i] -= pi/2
        #gradient shift rule
        gradient_D[i] = 0.5*(LHST(alpha1, alpha_W, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
                             -LHST(alpha2, alpha_W, alpha_W, shot_num, qubit_num, backend_sim, noise_model))
    return gradient_D

def gradient_descent(alpha_D, alpha_W, shot_num, qubit_num, backend_sim, noise_model, learning_rate, thresh_value):
    ##gradient descent of the cost function
    #learning rate determines the scale of change at each optimization step
    #thresh_value is the value to determine convergence and stop optimization
    #backend_sim refers to type_sim in funciton LHST
    costfunc = LHST(alpha_D, alpha_W, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
    print(costfunc)
    while(costfunc>thresh_value):    
        gradient_D = LHST_gradient_D(alpha_D, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
        gradient_W = LHST_gradient_W(alpha_D, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
        alpha_D = alpha_D - gradient_D*learning_rate
        alpha_W = alpha_W - gradient_W*learning_rate
        costfunc = LHST(alpha_D, alpha_W, alpha_W, shot_num, qubit_num, backend_sim, noise_model)
        print(costfunc)
        print(gradient_D, gradient_W)
    return alpha_D, alpha_W


def gradient_descent_curve(alpha_D_initial, alpha_W_initial, shot_num, qubit_num, backend_sim, noise_model, learning_rate, optim_step):
    ##generate gradient descent curve with a defined optimization step
    #learning rate determines the scale of change at each optimization step    
    #backend_sim refers to type_sim in funciton LHST
    #optim_step determines the number of optimization step
    costf = np.zeros(optim_step,dtype = float)   
    alpha_D = np.zeros((optim_step,len(alpha_D_initial)),dtype = float)
    alpha_W = np.zeros((optim_step,len(alpha_W_initial)),dtype = float)
    alpha_D[0,:] = alpha_D_initial.copy()
    alpha_W[0,:] = alpha_W_initial.copy()
    for i in range(optim_step):    
        costf[i] = LHST(alpha_D[i,:], alpha_W[i,:], alpha_W[i,:], shot_num, qubit_num, backend_sim, noise_model)
        gradient_D = LHST_gradient_D(alpha_D[i,:], alpha_W[i,:], shot_num, qubit_num, backend_sim, noise_model)
        gradient_W = LHST_gradient_W(alpha_D[i,:], alpha_W[i,:], shot_num, qubit_num, backend_sim, noise_model)
        if(i< optim_step-1):
            alpha_D[i+1,:] = alpha_D[i,:] - gradient_D*learning_rate
            alpha_W[i+1,:] = alpha_W[i,:] - gradient_W*learning_rate
            print(costf[i])
            print(gradient_D,gradient_W)
    return costf, alpha_D, alpha_W

def Fidelity_Test(alpha_D_1, alpha_W_1, alpha_D_2, alpha_W_2):
    ##simple overlap calculation between the states generated by two circuits
    circuit_1 = Bind_parameter_circuit(alpha_D_1, alpha_W_1, alpha_W_1)
    circuit_2 = Bind_parameter_circuit(alpha_D_2, alpha_W_2, alpha_W_2)    
    MES_reverse_generate(circuit_1, qubit_num) #MES_reverse is needed for constructing HST circuit
    job_1 = execute(circuit_1,backend = statevector_simulator)
    result_1 = job_1.result().get_statevector()
    MES_reverse_generate(circuit_2, qubit_num)
    job_2 = execute(circuit_2,backend = statevector_simulator)
    result_2 = job_2.result().get_statevector()
    result_1 = result_1.flatten()
    result_2 = result_2.flatten()
    # print('result1',result_1)
    # print('result2',result_2)
    value = 0
    for i in range(len(result_1)):
        value += abs(result_1[i]*(result_2[i].conjugate()))**2
    value = np.sqrt(value)
    return value

#construct the circuit for calculating VFF dynamics
circ_VFF = QuantumCircuit(QuantumRegister(1))
circ_VFF.append(W_circ.to_instruction(),[0])
circ_VFF.append(D_circ.to_instruction(),[0])
circ_VFF.append(Wdag_circ.to_instruction(),[0])

def VFF(alpha_D, alpha_W, deltat, N, shot_num, backend_sim):
    ##calculate dynamics by VFF
    #alpha_D and alpha_W are the optimal parameters
    #N*deltat is the total time length
    #shot_num is the number of measurements
    circuit = circ_VFF.bind_parameters({D_parameters[i]: N*alpha_D[i] for i in range(len(alpha_D))})
    circuit = circuit.bind_parameters({W_parameters[i]: alpha_W[i] for i in range(len(alpha_W))})
    circuit = circuit.bind_parameters({Wdag_parameters[i]: -alpha_W[i] for i in range(len(alpha_W))})
    circuit.measure_all()
    #job = execute(circuit,backend = qasm_simulator,shots=shot_num,coupling_map=coupling_map,
                      #basis_gates=basis_gates,
                      #noise_model=noise_model)
    job = execute(circuit,backend = qasm_simulator,shots=shot_num)
    result = job.result()
    counts = result.get_counts(circuit)
    return N*deltat, counts
    
def Trotterization_dynamic(deltat, N, shot_num, backend_sim, noise_model):
    ##calculate dynamics by Trotterization
    #backend_sim determines using noiseless or noisy simulator
    circ_T = QuantumCircuit(QuantumRegister(2))
    for i in range(N):
        circ_T.rz(-2*a[0]*deltat, 0)
        circ_T.ry(2*a[1]*deltat, 0)
        circ_T.rx(-2*a[2]*deltat, 0)
    circ_T.measure_all()
    if(backend_sim == 0):
        job = execute(circ_T,backend = qasm_simulator,shots=shot_num)
    if(backend_sim == 1):
        job = execute(circ_T,backend = qasm_simulator,shots=shot_num,
                      basis_gates=noise_model.basis_gates,
                      noise_model=noise_model)
    result = job.result()
    counts = result.get_counts(circ_T)
    return N*deltat, counts

def IBMQ_noise_model(str_backend):
    ##load noise model of quanutm conputers from IBMQ
    noise_simulator = provider.get_backend(str_backend)
    noise_model = NoiseModel.from_backend(noise_simulator)
    coupling_map = noise_simulator.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, basis_gates

def depolarizing_model(param):
    ##create depolarizing channel
    #param determines noise scale
    error_1 = depolarizing_error(param,1)
    error_2 = depolarizing_error(param,2)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['rz','sx','x','id'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
    return noise_model

def phase_damping_model(param):
    ##create phase damping channel
    #param determines noise scale    
    error = phase_damping_error(param)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error,['rz','sx','x','id'])
    return noise_model

def amplitude_damping_model(param):
    ##create amplitude damping channel
    #param determines noise scale     
    error = amplitude_damping_error(param)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error,['rz','sx','x','id'])
    return noise_model

def count_frequency(counts, component, shot_num):
    ## count the frequency of one possible measurement result
    #component: the possible measurement output
    if component in counts:
        value = counts[component]
    else:
        value = 0
    return float(value/shot_num)
    
def Trotter_curve(N, deltat, shot_num, backend_sim, noise_model):
    ##generate Trotterization dynamical curve, based on Trotterization_dynamic function   
    Trotter_time = np.zeros(N, dtype = float)
    Trotter = np.zeros([2,N], dtype = float)
    for i in range(N):
        t, Trotter_counts = Trotterization_dynamic(deltat, i, shot_num, backend_sim, noise_model)
        Trotter_time[i] = t
        Trotter[0,i] = count_frequency(Trotter_counts, '0', shot_num)
        Trotter[1,i] = count_frequency(Trotter_counts, '1', shot_num)
    return Trotter_time, Trotter

def VFF_curve(N, deltat, shot_num, alpha_D, alpha_W):
    ##generate VFF dynamical curve, based on VFF_curve function
    VFF_time = np.zeros(N, dtype = float)
    VFF_curve_value = np.zeros([2,N], dtype = float)
    for i in range(N):
        t, VFF_counts = VFF(alpha_D, alpha_W, deltat, i, shot_num, qasm_simulator)
        VFF_time[i] = t
        VFF_curve_value[0,i] = count_frequency(VFF_counts, '0', shot_num)
        VFF_curve_value[1,i] = count_frequency(VFF_counts, '1', shot_num)              
    return VFF_time, VFF_curve_value

def Fidelity_curve(curve_1, curve_2):
    ##generate the Fidelity curve between two dynamical curves
    Fidelity = curve_1*curve_2
    Fidelity = np.sqrt(Fidelity)
    Fidelity = np.sum(Fidelity, axis = 0)
    return Fidelity

def Hamiltonian_Hubbard(Ham_para):
    ##two-site Hubbard Hamiltonian
    H = (-Ham_para[0]*X^I)+(- Ham_para[0]*I^X)+(Ham_para[1]*Z^Z)
    return H

def Hamiltonian_onequb(a):
    ##single spin Hamiltonian
    H = -a[2]*X+a[1]*Y-a[0]*Z
    return H
    
def Hamiltonian_dynamic(N, deltat, Ham):
    ##calculate dynamics of single Hamiltionian with matrix mutiplication
    H = Ham
    H_curve = np.zeros([2,N],dtype = float)
    H_matrix = H.to_matrix().real
    for i in range(N):
        H_gate = HamiltonianGate(H_matrix, i*deltat)
        uni_matrix = H_gate.to_matrix()
        proba = uni_matrix*uni_matrix.conj()
        proba = proba[:,0]
        H_curve[:,i] = proba.real
    return H_curve  

def accumulate_infidelity(VFF_time, Fidelity):
    ##calculate acuumulated infidelity with respect to time
    Infidelity = np.ones(len(VFF_time), dtype = float)
    Infidelity = Infidelity - Fidelity
    Infidelity_accu = np.zeros(len(VFF_time), dtype = float)
    for i in range(len(VFF_time)):
        Infidelity_accu[i] = simps(Infidelity[:i+1],VFF_time[:i+1])
    return Infidelity_accu

def InFedlity_curve(curve_1, curve_2):
    ##calculate infidelity curve
    Fidelity = curve_1*curve_2
    Fidelity = np.sqrt(Fidelity)
    Fidelity = np.sum(Fidelity, axis = 0)
    Infidelity = np.ones(len(Fidelity), dtype = float)
    Infidelity = Infidelity - Fidelity
    return Infidelity

def Exp_curve(x,A,B,C):
    ##Exponential decay curve
    return A*np.exp(B*x)+C

def Optimization_extrpolation(optim_step, alpha_optimization):
    ##fit with Exp_curve
    popt, pcov = op.curve_fit(Exp_curve, optim_step, alpha_optimization,p0 = [0,0,1],maxfev=100000)
    return popt

#format of words used in plots
font1 = {
    'family':'Times New Roman',
    'weight': 'normal',
    'size': 18,
    }

font2 = {
    'family':'Times New Roman',
    'weight': 'normal',
    'size': 20,
    }

##Generate data and plot gradient descent curve
optim_step = 500
params = [10**(-5),10**(-4),10**(-3),0.01,0.02,0.03,0.05,0.06,0.08,0.1,0.2,0.3,0.5] #control params of noise model
repeat_num = 1
alpha_D_initial = alpha_D.copy()

alpha_W_initial = alpha_W.copy()

alpha_D = np.zeros([len(params), repeat_num, optim_step,len(alpha_D)],dtype = float)
alpha_W = np.zeros([len(params), repeat_num, optim_step,len(alpha_W)],dtype = float) 

costf = np.zeros([len(params), repeat_num, optim_step],dtype = float)

noise_model = [[] for i in range(len(params))]
for i in range(len(params)):
    noise_model[i] = phase_damping_model(params[i]) ##construct noise model
    
for i in range(len(params)):
    time_start = time.time()
    for j in range(repeat_num):
        costf[i,j,:], alpha_D[i,j,:,:], alpha_W[i,j,:,:] = gradient_descent_curve(alpha_D_initial, alpha_W_initial, shot_num, qubit_num, 2, noise_model[i], 1, optim_step)
    time_end = time.time()
    print('time cost of ideal simulator',time_end-time_start,'s')
    mean_costf = np.mean(costf[i,:,:], axis = 0)
    mean_alpha_D = np.mean(alpha_D[i,:,:,:], axis = 0)
    mean_alpha_W = np.mean(alpha_W[i,:,:,:], axis = 0)
    costf_exact = np.zeros(optim_step, dtype = float)
    for k in range(optim_step):
        costf_exact[k] = LHST(mean_alpha_D[k,:],mean_alpha_W[k,:],mean_alpha_W[k,:],shot_num,qubit_num, 0, noise_model[i])
    
    plt.figure(figsize = (12.0,9.0))
    plt.axes(yscale = 'log')
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()

    plt.plot(range(optim_step), mean_costf, label = str(params[i]), color = 'red', linestyle = '-', marker = 'None')
    plt.plot(range(optim_step), costf_exact, label = 'exact result of '+str(params[i]), color = 'red', linestyle = '--', marker = '.')
    plt.xlabel('optimization', fontsize = 22)
    plt.ylabel('cost function', fontsize = 22)
    plt.legend(prop = font1, loc = 'center right')
    figname = str(params[i])+'PhaseDamping'+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
    plt.savefig(figname)
    plt.show()


    dataname = str(params[i])+'PhaseDamping'+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.pkl'
    f = open(dataname,'wb')
    pickle.dump((a, deltat, optim_step, shot_num, repeat_num, costf[i,:,:], alpha_D[i,:,:,:], alpha_W[i,:,:,:], costf_exact),f)
    f.close() 
    
##Process data
alpha_D = np.zeros([len(params), repeat_num, optim_step,len(alpha_D)],dtype = float)
alpha_W = np.zeros([len(params), repeat_num, optim_step,len(alpha_W)],dtype = float) 

costf = np.zeros([len(params), repeat_num, optim_step],dtype = float)
mean_costf = np.zeros([len(params), optim_step],dtype = float)
mean_alpha_D = np.zeros([len(params), optim_step, len(alpha_D_initial)],dtype = float)
mean_alpha_W = np.zeros([len(params), optim_step, len(alpha_W_initial)],dtype = float) 
costf_exact = np.zeros([len(params), optim_step],dtype = float)

fit_alpha_D = np.zeros([len(params), len(alpha_D_initial)],dtype = float)
fit_alpha_W = np.zeros([len(params), len(alpha_W_initial)],dtype = float) 
fit_costf =  np.zeros(len(params),dtype = float) 

noise_model = [[] for i in range(len(params))]
for i in range(len(params)):
    noise_model[i] = phase_damping_model(params[i])
    
for i in range(len(params)):
    dataname = str(params[i])+'PhaseDamping'+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.pkl'
    f = open(dataname,'rb')
    mu, deltat, optim_step, shot_num, repeat_num, costf[i,:,:], alpha_D[i,:,:,:], alpha_W[i,:,:,:], costf_exact[i,:] = pickle.load(f)
    f.close() 
    mean_costf[i,:] = np.mean(costf[i,:,:], axis = 0)
    mean_alpha_D[i,:,:] = np.mean(alpha_D[i,:,:,:], axis = 0)
    mean_alpha_W[i,:,:] = np.mean(alpha_W[i,:,:,:], axis = 0)

#plot cost function curves    
plt.figure(figsize = (12.0,9.0))
plt.axes(yscale = 'log')
plt.tick_params(labelsize=20)
ax = plt.gca()
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right()

for i in range(len(params)):
    #plt.plot(range(optim_step), mean_costf[i,:], label = 'Depolarizing Error = '+str(params[i]), color = color[i], linestyle = '-', marker = 'None')
    plt.plot(range(optim_step), costf_exact[i,:], label = 'Error = '+str(params[i]), linestyle = '--', marker = '.')


plt.xlabel('optimization', fontsize = 22)
plt.ylabel('cost function', fontsize = 22)
plt.legend(prop = font1, loc = 'lower left')
figname = 'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
plt.savefig(figname)
plt.show()

#plot parameters' value versus noise strength
plt.figure(figsize = (12.0,9.0))
plt.axes(xscale = 'log')
plt.tick_params(labelsize=20)
plt.plot(params, mean_alpha_D[:,-1,0],label = 'γz', color = color[0], linestyle = '-', marker = '.')    
plt.plot(params, mean_alpha_W[:,-1,0],label = 'θz', color = color[1], linestyle = '-', marker = '.')    
plt.plot(params, mean_alpha_W[:,-1,1],label = 'θx', color = color[2], linestyle = '-', marker = '.')   
plt.xlabel('error strength', fontsize = 22)
plt.ylabel('parmeter value', fontsize = 22)
plt.legend(prop = font1, loc = 'center right')
figname = 'ParaDepolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
plt.savefig(figname)
plt.show()

#Plot parameters' value versus optimization step curve
for j in range(len(D_parameters)):
    plt.figure(figsize = (12.0,9.0))
    #plt.axes(yscale = 'log')
    plt.tick_params(labelsize=20)
    plt.grid()
    for i in range(len(params)):
        filter_lin = np.round(np.linspace(0,499,50))
        filter_lin = filter_lin.astype(int)
        y = mean_alpha_D[i,:,j]
        x = np.array(range(optim_step))
        plt.plot(x,y,label = 'Error = '+str(params[i]), linestyle = '-', marker = '.')    
    plt.xlabel('optimization', fontsize = 22)
    plt.ylabel(D_parameters[j].name, fontsize = 22)
    plt.legend(prop = font1, loc = 'center right')
    figname = 'ParaCurve'+D_parameters[j].name+'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
    plt.savefig(figname)
    plt.show()

for j in range(len(W_parameters)):
    plt.figure(figsize = (12.0,9.0))
    #plt.axes(yscale = 'log')
    plt.tick_params(labelsize=20)
    plt.grid()

    for i in range(len(params)):
        filter_lin = np.round(np.linspace(0,499,50))
        filter_lin = filter_lin.astype(int)
        y = mean_alpha_W[i,:,j]
        x = np.array(range(optim_step))
        popt = Optimization_extrpolation(x,mean_alpha_W[i,:,j])
        fit_alpha_W[i,j] = popt[2]
        plt.plot(x,y,label = 'Error = '+str(params[i]), linestyle = '-', marker = '.')       
    plt.xlabel('optimization', fontsize = 22)
    plt.ylabel(W_parameters[j].name, fontsize = 22)
    plt.legend(prop = font1, loc = 'center left')
    figname = 'ParaCurve'+W_parameters[j].name+'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
    plt.savefig(figname)
    plt.show()
 
# Fit the parameters' value versus optimization step curve with exponential decay function and plot the result
for i in range(len(params)):
    filter_lin = np.round(np.linspace(0,499,50))
    filter_lin = filter_lin.astype(int)
    x = np.array(range(optim_step))
    plt.figure(figsize = (12.0,9.0))
    plt.tick_params(labelsize=20)
    plt.grid()
    for j in range(len(D_parameters)):
        y = mean_alpha_D[i,:,j]
        popt = Optimization_extrpolation(x[100:300], y[100:300])
        y_fit = Exp_curve(x,popt[0],popt[1],popt[2])
        plt.plot(x[filter_lin],y[filter_lin],label = D_parameters[j].name, color = color[j], linestyle = '--', marker = '.')    
        plt.plot(x[filter_lin],y_fit[filter_lin],label = 'fitted'+D_parameters[j].name+'='+str(popt[2]), color = color[j], linestyle = '-', marker = 'None')   
    plt.title('Error = '+ str(params[i]))
    plt.xlabel('optimization', fontsize = 22)
    plt.ylabel('Value of parameters', fontsize = 22)
    plt.legend(prop = font2, loc = 'center right')
    figname = str(params[i])+'ParaCurveD'+'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
    plt.savefig(figname)
    plt.show()
    
for i in range(len(params)):
    filter_lin = np.round(np.linspace(0,499,50))
    filter_lin = filter_lin.astype(int)
    x = np.array(range(optim_step))
    plt.figure(figsize = (12.0,9.0))
    plt.tick_params(labelsize=20)
    plt.grid()
    for j in range(len(W_parameters)):
        y = mean_alpha_W[i,:,j]
        popt = Optimization_extrpolation(x[100:300], y[100:300])
        y_fit = Exp_curve(x,popt[0],popt[1],popt[2])
        plt.plot(x[filter_lin],y[filter_lin],label = W_parameters[j].name, color = color[j], linestyle = '--', marker = '.')    
        plt.plot(x[filter_lin],y_fit[filter_lin],label = 'fitted'+W_parameters[j].name, color = color[j], linestyle = '-', marker = 'None')   
    plt.title('Error = '+ str(params[i]))
    plt.xlabel('optimization', fontsize = 22)
    plt.ylabel('Value of parameters', fontsize = 22)
    plt.legend(prop = font2, loc = 'center right')
    figname = str(params[i])+'ParaCurveW'+'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
    plt.savefig(figname)
    plt.show()

##cost function of extrapolated parameters and cost function of original parameters
for j in range(len(D_parameters)):
    for i in range(len(params)):
        y = mean_alpha_D[i,:,j]
        x = np.array(range(optim_step))
        popt = Optimization_extrpolation(x[100:500],y[100:500])
        fit_alpha_D[i,j] = popt[2]

for j in range(len(W_parameters)):
    for i in range(len(params)):
        y = mean_alpha_W[i,:,j]
        x = np.array(range(optim_step))
        popt = Optimization_extrpolation(x[100:500],y[100:500])
        fit_alpha_W[i,j] = popt[2]
        
for i in range(len(params)):
    fit_costf[i] = LHST(fit_alpha_D[i,:],fit_alpha_W[i,:],fit_alpha_W[i,:],10000,1,0,0)

plt.figure(figsize = (12.0,9.0))
#plt.axes(xscale = 'log')
#plt.axes(yscale = 'log')
plt.tick_params(labelsize=20)
plt.xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(0)])
plt.loglog(params,costf_exact[:,-1],label = 'costf of original parameters',color = color[0], linestyle = '-', marker = '.')
plt.loglog(params,fit_costf,label = 'costf of fitted parameters',color = color[1],linestyle='--',marker = '.')
plt.xlabel('error strength', fontsize = 22)
plt.ylabel('cost function value', fontsize = 22)
plt.legend(prop = font2, loc = 'center left')
figname = 'CostfDepolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
plt.savefig(figname)
plt.show()

# # extrapolated parameters and parameters
# plt.figure(figsize = (12.0,9.0))
# plt.axes(xscale = 'log')
# plt.tick_params(labelsize=20)
# plt.xticks([10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(0)])
# plt.plot(params, mean_alpha_D[:,-1,0],label = 'γz', color = color[0], linestyle = '-', marker = '.')    
# plt.plot(params, mean_alpha_W[:,-1,0],label = 'θx', color = color[1], linestyle = '-', marker = '.')    
# plt.plot(params, mean_alpha_W[:,-1,1],label = 'θz', color = color[2], linestyle = '-', marker = '.')   
# plt.plot(params, fit_alpha_D[:,0],label = 'mitigated γz', color = color[0], linestyle = '--', marker = '.')    
# plt.plot(params, fit_alpha_W[:,0],label = 'mitigated θx', color = color[1], linestyle = '--', marker = '.')    
# plt.plot(params, fit_alpha_W[:,1],label = 'mitigated θz', color = color[2], linestyle = '--', marker = '.')   
# plt.xlabel('error strength', fontsize = 22)
# plt.ylabel('parmeters value', fontsize = 22)
# plt.ylim([0,1])
# plt.legend(prop = font2, loc = 'center left')
# figname = 'FitParaDepolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
# plt.savefig(figname)
# plt.show()

# # normal plot    
# plt.figure(figsize = (12.0,9.0))
# plt.axes(yscale = 'log')
# plt.tick_params(labelsize=20)
# ax = plt.gca()
# ax.yaxis.set_label_position('right')
# ax.yaxis.tick_right()

# for i in range(len(params)):
#     plt.plot(range(optim_step), mean_costf[i,:], label = 'Depolarizing Error = '+str(params[i]), color = color[i], linestyle = '-', marker = 'None')
#     plt.plot(range(optim_step), costf_exact[i,:], label = 'exact result of Error = '+str(params[i]), color = color[i], linestyle = '--', marker = '.')


# plt.xlabel('optimization', fontsize = 22)
# plt.ylabel('cost function', fontsize = 22)
# plt.legend(prop = font1, loc = 'center right')
# figname = 'Depolarizing_mu'+str(mu)+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
# plt.savefig(figname)
# plt.show()

##comparison of dynamical curves calculated by different methods
# N = 100

# H_curve = Hamiltonian_dynamic(100,deltat,Hamiltonian_onequb(a))
# VFF_time, Trotter = Trotter_curve(N, deltat, 10000, 0 ,None)
# #Trotter_Fidelity_H = Fidelity_curve(H_curve, Trotter)
# Trotter_InFidelity_H = InFedlity_curve(Trotter, H_curve)

# VFF_curve_value = np.zeros([len(params),2,N])
# VFF_Fidelity_H = np.zeros([len(params),N])
# VFF_InFidelity_H = np.zeros([len(params),N])
# for i in range(len(params)):
#     VFF_time, VFF_curve_value[i,:,:] = VFF_curve(N, deltat, 10000, mean_alpha_D[i,-1,:], mean_alpha_W[i,-1,:])
#     #VFF_Fidelity_H[i,:] = Fidelity_curve(H_curve, VFF_curve_value[i,:,:]) 
#     VFF_InFidelity_H[i,:] = InFedlity_curve(VFF_curve_value[i,:,:], H_curve)

# # plot dynamic curve
# plt.figure(figsize = (12.0,9.0))
# plt.tick_params(labelsize=20)

# #plt.plot(VFF_time, H_curve[0,:], label = 'Trotter without noise', color = 'black', linestyle = '-', marker = '.')
# plt.plot(VFF_time, Trotter[0,:], label = 'Trotter without noise', color = 'black', linestyle = '-', marker = '.')
# for i in range(len(params)):
#     plt.plot(VFF_time, VFF_curve_value[i,0,:], label = 'VFF Depolarizing Error = '+str(params[i]), color = color[i], linestyle = '-', marker = '.')
    

# plt.xlabel('t = N*deltat', fontsize = 22)
# plt.ylabel('P(0)', fontsize = 22)
# plt.legend(prop = font1, loc = 'center right')
# figname = 'P0_Depolarizing'+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
# plt.savefig(figname)
# plt.show()

# #plot Infidelity curve
# plt.figure(figsize = (12.0,9.0))
# plt.tick_params(labelsize=20)
# plt.axes(yscale='log')
# ax = plt.gca()
# ax.yaxis.set_label_position('right')
# ax.yaxis.tick_right()

# #plt.plot(VFF_time, Trotter_InFidelity_H, label = 'Trotter without noise', color = 'black', linestyle = '-', marker = '.')
# for i in range(len(params)):
#     plt.plot(VFF_time, VFF_InFidelity_H[i,:], label = 'VFF Depolarizing Error = '+str(params[i]), color = color[i], linestyle = '-', marker = '.')
    

# plt.xlabel('t = N*deltat', fontsize = 22)
# plt.ylabel('InFidelity', fontsize = 22)
# plt.legend(prop = font1, loc = 'center right')
# figname = 'InFidelity_Depolarizing'+'deltat'+str(deltat)+'opti'+str(optim_step)+'shot'+str(shot_num)+'.jpg'
# plt.savefig(figname)
# plt.show()

