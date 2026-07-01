import numpy as np
from numba import njit,cfunc,carray
import scipy.integrate as integrate
from numbalsoda import lsoda_sig,lsoda
import utils
import time

############################
### PARAMETER ASSERTIONS ###
############################

def assertParams(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):
    assert np.shape(muMatrix) == (Ns,Nr)
    assert np.shape(dTensor) == (Nr,Nr)
    assert np.shape(supplyVec) == (Nr,)
    assert np.shape(lVector) == (Ns,)

    assert np.all(supplyVec >= 0)
    assert delta > 0
    assert np.all(muMatrix >= 0)
    assert np.all(dTensor >= 0)

############################
### PARAMETER GENERATION ###
############################

def generate_lognormal(needed_mean,needed_sd,shape):
    lg_mean,lg_sd = utils.normal_to_lognormal(needed_mean,needed_sd)
    return np.random.lognormal(lg_mean,lg_sd,shape)

def generate_gamma(needed_mean,needed_sd,shape):
    gamma_shape,gamma_scale = needed_mean**2/needed_sd**2,needed_sd**2/needed_mean
    return np.random.gamma(gamma_shape,gamma_scale,shape)

def generate_random_choice(needed_mean,needed_sd,shape,randomFlag):
    high_rand = 2 if randomFlag else 1
    randint = np.random.randint(0,high_rand)
    paramMatrix = generate_lognormal(needed_mean,needed_sd,shape) if randint == 0 else generate_gamma(needed_mean,needed_sd,shape)
    return paramMatrix

def generate_mu(Ns,Nr,mean,sd):
    muMatrix = generate_lognormal(mean,sd,(Ns,Nr))/Nr
    return muMatrix

def generate_dTensor(Nr):
    dTensor = np.random.uniform(0,1/(Nr-1),(Nr,Nr))
    dTensor = dTensor - np.diag(np.diag(dTensor))
    return dTensor

def generate_lVector(Ns,min_val,max_val):
    lVector = np.random.uniform(min_val,max_val,Ns)
    return lVector

########################
### eEO INTERACTIONS ###
########################

def with_affinities(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):

    @njit
    def sigmaVec_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):
        sigmaVec = delta*(supplyVec - rVec)
        return sigmaVec

    @njit
    def fMatrix_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):
        fMatrix = - (muMatrix*rVec/(1+rVec)).T + (dTensor@((muMatrix * rVec/(1+rVec)).T*lVector))    
        return fMatrix

    @njit
    def sMatrix_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):
        sMatrix = (muMatrix.T*(1-lVector)).T/(1+rVec)**2
        return sMatrix
    
    @njit
    def eo_interaction_params(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):

        sigmaVec = sigmaVec_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        fMatrix = fMatrix_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        sMatrix = sMatrix_fn(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)

        growthVec = np.dot(sMatrix,sigmaVec) 
        interactionMatrix = np.dot(sMatrix,fMatrix)
        return growthVec,interactionMatrix

    return eo_interaction_params(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)

################
### DYNAMICS ###
################

def make_lsoda_func_dtx_dynamics_with_aff(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr):
    @cfunc(lsoda_sig)
    def dtx_dynamics_lsoda(t, x, dx, p):
        x_ = carray(x, (Ns+Nr,))
        dx_ = carray(dx, (Ns+Nr,))
        populations = x_[:Ns]
        resources = x_[Ns:]

        uptakeMatrix = (muMatrix.T * populations).T * resources/(1+resources)
        resourceUsage = np.sum(uptakeMatrix,axis=0)          
        leakage = np.sum(np.dot(dTensor,(uptakeMatrix.T*lVector)),axis=1)


        dx_[:Ns] = (np.sum(uptakeMatrix,axis=1)*(1-lVector) - delta*populations )   
        dx_[Ns:] = delta*(supplyVec[:Nr] - resources) - resourceUsage + leakage

        dx_[:Ns][populations < 1e-20] = 0
    return dtx_dynamics_lsoda

###############
### SOLVERS ###
###############

def main_simulate_fn(Nr,Ns,supplyVec,delta,species_params,maxTime):
    muMean,muSd,lMin,lMax = species_params
    muMatrix = generate_mu(Ns,Nr,muMean,muSd)
    dTensor = generate_dTensor(Nr)
    lVector = generate_lVector(Ns,lMin,lMax)
    assertParams(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)

    initialConditions = np.concatenate((np.full(Ns,1),supplyVec))
    t = np.linspace(0,maxTime,1000)

    dynamics_lsoda = make_lsoda_func_dtx_dynamics_with_aff(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
    funcptr = dynamics_lsoda.address
    usol_temp,success_temp = lsoda(funcptr, initialConditions.flatten(), np.linspace(0,0.1,10))
    usol, success = lsoda(funcptr, initialConditions.flatten(), t,rtol=1e-6,atol=1e-8)
    final_tpoints = usol[-1,:]
    del usol,usol_temp
    rVec = final_tpoints[Ns:]
    eo_growth,eo_inter = with_affinities(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)

    return muMatrix,dTensor,lVector,eo_growth,eo_inter,final_tpoints

def multiple_runs(Nr,Ns,supplyVec,delta,species_params,numRuns,maxTime,seed):
    np.random.seed(seed)

    muMatrices = np.zeros((numRuns,Ns,Nr))
    dMatrices = np.zeros((numRuns,Nr,Nr))
    lVectors = np.zeros((numRuns,Ns))
    eoGrowths = np.zeros((numRuns,Ns))
    eoInters = np.zeros((numRuns,Ns,Ns))
    chemostatSolutions = np.zeros((numRuns,Ns+Nr))
    rescaledEOInters = np.zeros(eoInters.shape)
    supplyVecArray = np.zeros((numRuns,Nr))
    for i in range(numRuns):
        muMatrices[i],dMatrices[i],lVectors[i],eoGrowths[i],eoInters[i],chemostatSolutions[i] = main_simulate_fn(Nr,Ns,supplyVec,delta,species_params,maxTime)
        rescaledEOInters[i] = (eoInters[i]/eoGrowths[i][:,np.newaxis]) / np.diag(eoInters[i]/eoGrowths[i][:,np.newaxis])
        supplyVecArray[i] = supplyVec
    return muMatrices,dMatrices,lVectors,supplyVecArray,eoGrowths,eoInters,rescaledEOInters,chemostatSolutions


def rescale_interactions(interactionMatrix, growthVec):
    bjj = np.diag(interactionMatrix)
    return (interactionMatrix * growthVec[None, :]) / (bjj[None, :] * growthVec[:, None])

