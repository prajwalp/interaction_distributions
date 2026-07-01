import numpy as np
from numbalsoda import lsoda

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cr_model
Ns = 100
Nr = 100

delta = 0.1

maxTime = 10000
t = np.linspace(0,maxTime,1000)

supplyVec = np.zeros(Nr)
supplyVec[:Nr//2] = 100


muMean,muSd = 1,1  ## change muSd to 10 for different CV
muMatrix = cr_model.generate_mu(Ns,Nr,muMean,muSd)
dTensor = cr_model.generate_dTensor(Nr)

num_l_values = 3
reps_l = 1
generated_l_vectors = np.zeros((num_l_values,reps_l,Ns))
growth_rates_matrix = np.zeros((num_l_values,reps_l,Ns))
interaction_matrices = np.zeros((num_l_values,reps_l,Ns,Ns))
rescaled_interaction_matrices = np.zeros((num_l_values,reps_l,Ns,Ns))
population_matrices = np.zeros((num_l_values,reps_l,Ns))

minL = 0
maxL = 0.4

supplyVec = np.zeros(Nr)
supplyVec[:Nr//2] = 10

min_l_vals = np.zeros(num_l_values)
max_l_vals = np.zeros(num_l_values)

for i in range(num_l_values):
    max_l_vals[i] = (2*i+0.1)*(maxL-minL)/num_l_values
    min_l_vals[i] = (2*i)*(maxL-minL)/num_l_values if i >= 1 else 0

for i in range(num_l_values):
    lmin = min_l_vals[i]
    lmax = max_l_vals[i]   
    assert lmax <= 1
    for j in range(reps_l):
        initialConditions = np.concatenate((np.full(Ns,1),supplyVec))
        lVector = cr_model.generate_lVector(Ns,lmin,lmax)

        dtx_dynamics_lsoda = cr_model.make_lsoda_func_dtx_dynamics_with_aff(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        funcptr = dtx_dynamics_lsoda.address
        usol_temp,success_temp = lsoda(funcptr, initialConditions.flatten(), np.linspace(0,0.1,10))
        usol, success = lsoda(funcptr, initialConditions.flatten(), t,rtol=1e-6,atol=1e-8)
        final_tpoints = usol[-1,:]
        rVec = final_tpoints[Ns:]

        tdepGrowthEnd,tdepInterEnd = cr_model.with_affinities(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        generated_l_vectors[i,j,:] = lVector
        population_matrices[i,j,:] = final_tpoints[:Ns]
        growth_rates_matrix[i,j,:] = tdepGrowthEnd
        interaction_matrices[i,j,:,:] = tdepInterEnd
        rescaled_interaction_matrices[i,j] = cr_model.rescale_interactions(tdepInterEnd, tdepGrowthEnd)
        surv_pop = np.where(final_tpoints[:Ns] > 0.01)[0]
    print(i)


np.savez("../data/figures/fig2/new_rescaled_interactions_N-cv-1.npz",
         generated_l_vectors=generated_l_vectors,
         growth_rates_matrix=growth_rates_matrix,
         interaction_matrices=interaction_matrices,
         rescaled_interaction_matrices=rescaled_interaction_matrices,
         population_matrices=population_matrices,
         min_l_vals=min_l_vals,
         max_l_vals=max_l_vals)