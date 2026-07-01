import numpy as np

from numba import njit,cfunc,carray
from numbalsoda import lsoda_sig,lsoda


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cr_model
import informed_glv
import utils


def compute_theory_matrix(muMatrix):
    Nr = muMatrix.shape[1]
    mean_mu = np.mean(muMatrix)
    sm_mu = np.mean(muMatrix**2)
    tm_mu = np.mean(muMatrix**3)

    self_limitation = Nr*sm_mu

    mean_interaction = Nr * mean_mu**2
    var_interaction = Nr * (sm_mu**2 - mean_mu**4)
    tc_interaction = Nr*(tm_mu**2 - 3*mean_mu**2*sm_mu**2 + 2*mean_mu**6)
    return mean_interaction, var_interaction, tc_interaction

def normal_to_lognormal(mean,sd):
    mu = np.round(np.log(mean**2/np.sqrt(mean**2 + sd**2)),8)
    sigma = np.round(np.sqrt(np.log(1 + sd**2/mean**2)),8)
    return mu,sigma

def compute_cumulants(matrix):
    mean = np.mean(matrix)
    var = np.var(matrix)
    tc = np.mean((matrix - mean)**3)
    return mean, var, tc

def compute_ipr_simulation(muMatrix):
    return np.mean(np.mean(muMatrix**4,axis=1) / np.mean(muMatrix**2,axis=1)**2/ muMatrix.shape[1])

########################
### COMPETITIVE SKEW ###
########################

def compute_crossfeeding_skew(muMatrix,dTensor,lVector):
    Nr = muMatrix.shape[1]
    l = np.mean(lVector)

    m1r = np.mean(muMatrix)
    m2r = np.mean(muMatrix**2)
    m3r = np.mean(muMatrix**3)
    m1d = np.mean(dTensor)
    m2d = np.mean(dTensor**2)

    m1a = Nr*m1r**2
    m2a = Nr*(Nr-1)*m1r**4 + Nr*m2r**2
    m3a = Nr*(Nr-1)*(Nr-2)*m1r**6 + 3*Nr*(Nr-1)*m1r**2*m2r**2 + Nr*m3r**2

    m1b = Nr*(Nr-1)*m1r**2 *m1d
    m2b = Nr*(Nr-1)*m2r**2*m2d + 2*Nr*(Nr-1)*(Nr-2)*m1r**2*m1d**2*m2r + Nr*(Nr-1)*(Nr-2)*(Nr-3)*m1r**4*m1d**2

    m1ab = Nr*(Nr-1)*(Nr-3) *m1r**4 *m1d + 2*Nr*(Nr-1)*m1r**2 *m2r*m1d
    m1a2b = 2*Nr*(Nr-1) *m3r *m2r * m1r*m1d + 2*Nr*(Nr-1)*m2r**2 *m1r**2*m1d + 4*Nr*(Nr-1)*(Nr-2)*m2r*m1r**4*m1d + Nr*(Nr-1)*(Nr-2)*(Nr-3)*m1r**6 *m1d

    k1a = m1a
    k2a = m2a - m1a**2
    k3a = m3a - 3*m1a*m2a + 2*m1a**3
    k2b = m2b - m1b**2
    covab = m1ab - m1a*m1b
    covaba = m1a2b - m1ab*m1a
    
    mean_correction = m1a+m1b
    var_correction = k2a+covab
    tc_correction = k3a+covaba-m1b*k2a - m1a*k2b

    mean_interaction = k1a - l*mean_correction
    var_interaction = k2a - 2*l*var_correction
    tc_interaction = k3a - 3*l*tc_correction
    return mean_interaction, var_interaction, tc_interaction
    

def macarthur_competitive(rVec,muMatrix,supplyVec,delta,Ns,Nr):

    @njit
    def sigmaVec_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        sigmaVec = delta*(supplyVec - rVec)
        return sigmaVec

    @njit
    def fMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        fMatrix = - (muMatrix*rVec/(1+rVec)).T 
        return fMatrix

    @njit
    def sMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        sMatrix = muMatrix/(1+rVec)**2
        return sMatrix
    
    @njit
    def eo_interaction_params(rVec,muMatrix,supplyVec,delta,Ns,Nr):

        sigmaVec = sigmaVec_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)
        fMatrix = fMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)
        sMatrix = sMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)

        growthVec = np.dot(sMatrix,sigmaVec) 
        interactionMatrix = np.dot(sMatrix,fMatrix)
        return growthVec,interactionMatrix

    return eo_interaction_params(rVec,muMatrix,supplyVec,delta,Ns,Nr)


Ns = 100
Nr = 100
mean_mu = 1
theory_sd_mu = np.round(np.linspace(1,10,20),3)

nreps = 100
interaction_cumulants = np.zeros((len(theory_sd_mu), nreps,3))
iprs = np.zeros((len(theory_sd_mu), nreps))
for i, sd in enumerate(theory_sd_mu):
    for reps in range(nreps):
        resc_mean,resc_sd = mean_mu/Nr, sd/Nr
        lg_mean, lg_sd = np.round(normal_to_lognormal(mean_mu, sd),3)
        muMatrix = np.random.lognormal(lg_mean,lg_sd, size=(Ns,Nr))/Nr
        interaction_cumulants[i,reps, :] = compute_theory_matrix(muMatrix)
        

supplyVec = np.full(Nr,5)
delta = 0.1
t = np.linspace(0,5000,1000)
initialConditions = np.concatenate((np.ones(Ns),supplyVec))
sd_mu = np.round(np.linspace(0.8,10,10),5)

simulation_reps = 50
simulation_cumulants = np.zeros((len(sd_mu), simulation_reps,3))
full_interaction_cumulants = np.zeros((len(sd_mu), simulation_reps,3))
non_diag_cumulants = np.zeros((len(sd_mu), simulation_reps,3))
simulation_iprs = np.zeros((len(sd_mu), simulation_reps))

for i, sd in enumerate(sd_mu):
    
    for reps in range(simulation_reps):
        muMatrix = np.random.lognormal(*normal_to_lognormal(mean_mu, sd), size=(Ns,Nr))/Nr
        comp_dynamics_lsoda = cr_model.make_lsoda_func_dtx_dynamics_with_aff(muMatrix,np.zeros((Nr,Nr)),np.zeros(Ns),supplyVec,delta,Ns,Nr)
        funcptr = comp_dynamics_lsoda.address
        usol, success = lsoda(funcptr, initialConditions.flatten(), t,atol=1e-10,rtol=1e-9)
        fail_count = 0
        while not success:
            print(f"Simulation failed for sd={sd} at rep={reps}")
            usol, success = lsoda(funcptr, initialConditions.flatten(), t,atol=1e-11,rtol=1e-10)
            fail_count += 1
            if fail_count > 10:
                print("Too many failures, breaking out of loop.")
                break
        rVec = usol[-1,:][Ns:]

        tdepGrowthEnd,tdepInterEnd = cr_model.with_affinities(rVec,muMatrix,np.zeros((Nr,Nr)),np.zeros(Ns),supplyVec,delta,Ns,Nr)
        rescaled_inters = (tdepInterEnd / tdepGrowthEnd [:,None]) / np.diag(tdepInterEnd / tdepGrowthEnd [:,None])
        non_diag_inters = rescaled_inters[rescaled_inters != 1]
        simulation_cumulants[i,reps,:] = compute_cumulants(non_diag_inters)
        full_interaction_cumulants[i,reps,:] = compute_cumulants(tdepInterEnd.flatten())
        non_diag_cumulants[i,reps,:] = compute_cumulants(tdepInterEnd[~np.eye(tdepInterEnd.shape[0],dtype=bool)].flatten())
        simulation_iprs[i, reps] = compute_ipr_simulation(muMatrix)

np.save("../data/figures/fig2/competitive_skew.npy", {
    "sd_mu": sd_mu,
    "simulation_cumulants": simulation_cumulants,
    "theory_sd_mu": theory_sd_mu,
    "theory_cumulants": interaction_cumulants})


#########################
### CROSSFEEDING SKEW ###
#########################

Ns = 100
Nr = 100
mean_mu = 1
sd_mu = 1 
lValues = np.linspace(0,0.3,21)

nreps = 100
interaction_cumulants_crossfeeding = np.zeros((len(lValues), nreps,3))

for reps in range(nreps): 
    dTensor = np.random.uniform(0,1/(Nr-1), size=(Nr,Nr))
    dTensor[np.eye(dTensor.shape[0],dtype=bool)] = 0

    muMatrix = np.random.lognormal(*normal_to_lognormal(mean_mu, sd_mu), size=(Ns,Nr))/Nr
    for i, lVal in enumerate(lValues):                               
            interaction_cumulants_crossfeeding[i,reps, :] += compute_crossfeeding_skew(muMatrix,dTensor,lVal)

supplyVec = np.zeros(Nr)
supplyVec[:Nr//2] = 10
delta = 0.1
initialConditions = np.concatenate((np.ones(Ns),supplyVec))
t = np.linspace(0,5000,1000)
lMinVals = np.round(np.linspace(0,0.3,10),3)
lMaxVals = lMinVals + 0.02

simulation_reps = 50
simulation_cumulants = np.zeros((len(lMinVals), simulation_reps,3))
full_interaction_cumulants = np.zeros((len(lMinVals), simulation_reps,3))
non_diag_cumulants = np.zeros((len(lMinVals), simulation_reps,3))
simulation_iprs = np.zeros((len(lMinVals), simulation_reps))
for reps in range(simulation_reps):
    muMatrix = np.random.lognormal(*normal_to_lognormal(mean_mu, sd_mu), size=(Ns,Nr))/Nr
    dTensor = np.random.uniform(0,1/(Nr-1), size=(Nr,Nr))
    dTensor[np.eye(dTensor.shape[0],dtype=bool)] = 0
    for lID in range(len(lMinVals)):        
        lVector = np.random.uniform(lMinVals[lID], lMaxVals[lID], size=Ns)

        cf_dynamics_lsoda = cr_model.make_lsoda_func_dtx_dynamics_with_aff(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        funcptr = cf_dynamics_lsoda.address
        usol, success = lsoda(funcptr, initialConditions.flatten(), t,atol=1e-10,rtol=1e-9)
        fail_count = 0
        while not success:
            print(f"Simulation failed for sd={sd} at rep={reps}")
            usol, success = lsoda(funcptr, initialConditions.flatten(), t,atol=1e-11,rtol=1e-10)
            fail_count += 1
            if fail_count > 10:
                print("Too many failures, breaking out of loop.")
                break
        rVec = usol[-1,:][Ns:]

        tdepGrowthEnd,tdepInterEnd = cr_model.with_affinities(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        rescaled_inters = (tdepInterEnd / tdepGrowthEnd [:,None]) / np.diag(tdepInterEnd / tdepGrowthEnd [:,None])
        non_diag_inters = rescaled_inters[rescaled_inters != 1]
        simulation_cumulants[lID,reps,:] = compute_cumulants(non_diag_inters)
        full_interaction_cumulants[lID,reps,:] = compute_cumulants(tdepInterEnd.flatten())
        non_diag_cumulants[lID,reps,:] = compute_cumulants(tdepInterEnd[~np.eye(tdepInterEnd.shape[0],dtype=bool)].flatten())
        simulation_iprs[lID, reps] = compute_ipr_simulation(muMatrix)
simulation_skew = simulation_cumulants[:,:, 2] / simulation_cumulants[:, :,1]**(3/2)

np.save("../data/figures/fig2/crossfeeding_skew.npy", {
    "lValues": lValues,
    "lMinVals": lMinVals,
    "lMaxVals": lMaxVals,
    "theory_cumulants": interaction_cumulants_crossfeeding,
    "simulation_cumulants": simulation_cumulants})