import numpy as np

from multiprocessing import Pool, cpu_count

import time, sys, os
from numbalsoda import lsoda

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cr_model
import informed_glv
import utils

def run_leakage(params):
    Ns,Nr,supplyVec,delta,muMean,muSd,lMin,lMax,chemostat_trials,n_glv_trials,randomseed  = params
    np.random.seed(randomseed)
    corr_fit_vals =  np.zeros((chemostat_trials,8))
    lg_fit_vals = np.zeros((chemostat_trials,5))
    gauss_fit_vals = np.zeros((chemostat_trials,2))

    correlated_results = np.zeros((n_glv_trials,3))
    lognormal_results = np.zeros((n_glv_trials,3))
    gaussian_results = np.zeros((n_glv_trials,3))
    chemostat_results = np.zeros((chemostat_trials,2))

    initialConditions = np.concatenate((np.full(Ns,0.1),supplyVec))
    t = np.linspace(0, 10000, 1000)

    for chemID in range(chemostat_trials):
        muMatrix = cr_model.generate_mu(Ns,Nr,muMean,muSd)
        dTensor = cr_model.generate_dTensor(Nr)
        lVector = cr_model.generate_lVector(Ns,lMin,lMax)

        cf_dynamics_lsoda = cr_model.make_lsoda_func_dtx_dynamics_with_aff(muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        funcptr = cf_dynamics_lsoda.address
        usol, success = lsoda(funcptr, initialConditions.flatten(), t,rtol=1e-6,atol=1e-8)
        final_tpoints = usol[-1,:]
        rVec = final_tpoints[Ns:Ns+Nr]
        surv_num = np.where(final_tpoints[:Ns] > 1e-3)[0]

        tdepGrowthEnd,tdepInterEnd = cr_model.with_affinities(rVec,muMatrix,dTensor,lVector,supplyVec,delta,Ns,Nr)
        rescaled_inter = cr_model.rescale_interactions(tdepInterEnd,tdepGrowthEnd)

        gauss_fit_vals[chemID] = np.mean(rescaled_inter[rescaled_inter != 1]),np.std(rescaled_inter[rescaled_inter != 1])
        lg_fit_vals[chemID] = informed_glv.compute_lg_fit_params(rescaled_inter,Ns)
        corr_fit_vals[chemID] = informed_glv.compute_correlation_fit_params(rescaled_inter,Ns)
        
        chemostat_results[chemID]  = surv_num.size,utils.compute_shannon(final_tpoints[:Ns])
        del usol

    mean_corr_fit = np.nanmean(corr_fit_vals,axis=0)
    mean_lg_fit = np.nanmean(lg_fit_vals,axis=0)
    mean_gauss_fit = np.nanmean(gauss_fit_vals,axis=0)

        

    for glvID in range(n_glv_trials):
        correlated_interactions = informed_glv.generate_correlated_matrix(mean_corr_fit,Ns)
        lognormal_interactions = informed_glv.generate_lg_matrix(mean_lg_fit,Ns)
        gaussian_interactions = informed_glv.generate_gaussian_matrix(mean_gauss_fit,Ns)

        correlated_results[glvID] = informed_glv.simulate_stability(correlated_interactions,10000,Ns)
        lognormal_results[glvID] = informed_glv.simulate_stability(lognormal_interactions,10000,Ns)
        gaussian_results[glvID] = informed_glv.simulate_stability(gaussian_interactions,10000,Ns)

    print(f"Finished {lMin} - {lMax}")
    return chemostat_results,correlated_results,lognormal_results,gaussian_results

def multiple_iterator(iterable):
    with Pool(cpu_count()) as p:
        results = p.map(run_leakage,iterable)
    return results

if __name__ == '__main__':
    print(time.ctime())
    
    NsList = np.array([60,100,140,180,220])
    NrList = np.array(NsList*1.75).astype(int)


    for nID in range(len(NsList)):
        Ns = NsList[nID]
        Nr = NrList[nID]
        
        delta = 0.1

        print(Ns)
        muMean,muSd = 1,10

        minL = 0.2
        maxL = 0.6
        num_l_values = 4
        lMinVals = np.zeros(num_l_values)
        lMaxVals = np.zeros(num_l_values)

        for i in range(num_l_values):
            lMaxVals[i] = minL + (i+2)*(maxL-minL)/num_l_values
            lMinVals[i] = minL + ((i)*(maxL-minL)/num_l_values if i >= 1 else 0)
            # lMinVals[i] = 0.00

        print(lMinVals,lMaxVals)

        randomSeeds = np.random.randint(0,100000,num_l_values)

        chemostat_trials = 50
        n_glv_trials = 100

        supplyVec = np.zeros(Nr)
        supplyVec[:Nr//4] = 20
        species_params = [[Ns,Nr,supplyVec,delta,muMean,muSd,lMinVals[i],lMaxVals[i],chemostat_trials,n_glv_trials,randomSeeds[i]] for i in range(num_l_values)]

        TSTARTTOTAL = time.time()
        functionResults = multiple_iterator(species_params)

        print(f"Total time: {time.time()-TSTARTTOTAL}\n")

        saveFolder = "data/predictability/"
        os.makedirs(saveFolder, exist_ok=True)
        utils.save_compressed_pickle(saveFolder+"glv_predictability_data_N-"+str(Ns),functionResults)
        utils.save_compressed_pickle(saveFolder+"glv_predictability_params_N-"+str(Ns),species_params)

    print(time.ctime())
