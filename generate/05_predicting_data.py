import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.integrate as integrate
import scipy.optimize as optimize
from numbalsoda import lsoda
import pandas as pd
import scipy.stats as stats
import itertools
import colormaps as cmaps
import seaborn as sns

from numba import njit,cfunc,carray
from numbalsoda import lsoda_sig,lsoda

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cr_model
import utils


@numba.njit
def glv(t,y,inter):
    ydot = y*(1 - inter@y)
    ydot[y<1e-20] = 0
    ydot[y>1e12] = 0
    return ydot

def compute_interactions(rVec,muMatrix,supplyVec,delta,Ns,Nr):
    def sigmaVec_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        sigmaVec = delta*(supplyVec - rVec)
        return sigmaVec

    def fMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        fMatrix = - (muMatrix*rVec).T    
        return fMatrix

    def sMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr):
        sMatrix = muMatrix
        return sMatrix
    
    def eo_interaction_params(rVec,muMatrix,supplyVec,delta,Ns,Nr):

        sigmaVec = sigmaVec_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)
        fMatrix = fMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)
        sMatrix = sMatrix_fn(rVec,muMatrix,supplyVec,delta,Ns,Nr)

        growthVec = np.dot(sMatrix,sigmaVec) 
        interactionMatrix = np.dot(sMatrix,fMatrix)
        return growthVec,interactionMatrix

    return eo_interaction_params(rVec,muMatrix,supplyVec,delta,Ns,Nr)

@numba.njit
def competitve_cr(t,y,muMatrix,delta,supply,Ns,Nr):
    npop = y[:Ns]
    rpop = y[Ns:]

    ndot = npop*(np.sum(muMatrix * rpop,axis=1) - delta)
    rdot = delta*(supply - rpop) - rpop*np.sum(muMatrix.T * npop,axis=1)
    ndot[npop<1e-20] = 0

    return np.concatenate((ndot,rdot))


def analyze_interactions(non_diag_resc):
    pos_frac = np.where(non_diag_resc < 0 )[0].size / non_diag_resc.size
    if(pos_frac > 0):
        pos_inters =  -non_diag_resc[non_diag_resc < 0]
        log_pos_mean,log_pos_std = np.mean(np.log(pos_inters)), np.std(np.log(pos_inters))
    else:
        log_pos_mean,log_pos_std = 0,0
    neg_inters = non_diag_resc[non_diag_resc > 0]
    log_neg_mean,log_neg_std = np.mean(np.log(neg_inters)), np.std(np.log(neg_inters))
    return log_pos_mean,log_pos_std,log_neg_mean,log_neg_std,pos_frac

def generate_lognormal(log_pos_mean,log_pos_std,log_neg_mean,log_neg_std,pos_frac,Ns):
    log_normal_inters = np.zeros((Ns,Ns))
    for i in range(Ns):
        for j in range(Ns):
            if(i != j):
                if(np.random.rand() < pos_frac):
                    log_normal_inters[i,j] = -np.random.lognormal(log_pos_mean,log_pos_std)
                else:
                    log_normal_inters[i,j] = np.random.lognormal(log_neg_mean,log_neg_std)
    np.fill_diagonal(log_normal_inters,1)
    return log_normal_inters

def generate_gaussian(mean,sd,Ns):
    inter_matrix = np.zeros((Ns,Ns))
    for i in range(Ns):
        for j in range(Ns):
            if(i != j):
                inter_matrix[i,j] = np.random.normal(mean,sd)
    np.fill_diagonal(inter_matrix,1)
    return inter_matrix

def generate_needed_matrices(rescaled_inter_matrix,needed_Ns):
    Ns = needed_Ns
    non_diag_resc = rescaled_inter_matrix[np.where(rescaled_inter_matrix != 1)]
    lognormal_params = analyze_interactions(non_diag_resc)
    gaussian_mean,gaussian_sd = np.mean(non_diag_resc), np.std(non_diag_resc)
    lognormal_interactions = generate_lognormal(*lognormal_params,Ns=Ns)
    gaussian_interactions = generate_gaussian(gaussian_mean,gaussian_sd,Ns=Ns)
    return gaussian_interactions,lognormal_interactions

#####################
### DUCKWEED DATA ###
#####################

full_pops = pd.read_csv("../data/interaction_data/ishizawa_et_al/full_comm_pops.csv").to_numpy()
pops = np.mean(full_pops,axis=0)
drop_one_pops = pd.read_csv("../data/interaction_data/ishizawa_et_al/drop_one_pops.csv").to_numpy()

inters_chain = pd.read_csv("../data/interaction_data/ishizawa_et_al/chains_upto_four.csv").to_numpy()[:,1:]
mean_inters = np.mean(inters_chain,axis=0)
sd_inters = np.std(inters_chain,axis=0)
cov_sd = np.diag(sd_inters**2)
inter_matrix = np.reshape(np.array(mean_inters),(7,7)).astype(float)
rescaled_inter_matrix = cr_model.rescale_interactions(inter_matrix, np.ones(inter_matrix.shape[0]))

maxTime = 1000
n_trials = 1000

full_comm_size = 7
doo_comm_size = 6

gaussian_diversities_full = np.zeros(n_trials)
lognormal_diversities_full = np.zeros(n_trials)

gaussian_diversities_doo = np.zeros(n_trials)
lognormal_diversities_doo = np.zeros(n_trials)

for trialID in range(n_trials):
    cur_rescaled_inter_matrix = rescaled_inter_matrix.copy()
    init_cond = np.full(full_comm_size,0.1)
    gaussian_approx,lognormal_approx = generate_needed_matrices(cur_rescaled_inter_matrix,full_comm_size) 
    gaussian_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(gaussian_approx,), method="LSODA")
    lognormal_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(lognormal_approx,), method="LSODA")

    if(gaussian_soln.y[:,-1] < 1e8).all():
        gaussian_shannon = utils.compute_shannon(gaussian_soln.y[:,-1])
    else:
        gaussian_shannon = -1
    if(lognormal_soln.y[:,-1] < 1e8).all():
        lognormal_shannon = utils.compute_shannon(lognormal_soln.y[:,-1])
    else:
        lognormal_shannon = -1
    gaussian_diversities_full[trialID] = gaussian_shannon
    lognormal_diversities_full[trialID] = lognormal_shannon


    init_cond = np.full(doo_comm_size,0.1)
    gaussian_approx,lognormal_approx = generate_needed_matrices(cur_rescaled_inter_matrix,doo_comm_size)
    gaussian_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(gaussian_approx,), method="LSODA")
    lognormal_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(lognormal_approx,), method="LSODA")

    if(gaussian_soln.y[:,-1] < 1e8).all():
        gaussian_shannon = utils.compute_shannon(gaussian_soln.y[:,-1])
    else:
        gaussian_shannon = -1
    if(lognormal_soln.y[:,-1] < 1e8).all():
        lognormal_shannon = utils.compute_shannon(lognormal_soln.y[:,-1])
    else:
        lognormal_shannon = -1
    gaussian_diversities_doo[trialID] = gaussian_shannon
    lognormal_diversities_doo[trialID] = lognormal_shannon


gaussian_diversities_full = gaussian_diversities_full[gaussian_diversities_full > 0]
lognormal_diversities_full = lognormal_diversities_full[lognormal_diversities_full > 0]
gaussian_diversities_doo = gaussian_diversities_doo[gaussian_diversities_doo > 0]
lognormal_diversities_doo = lognormal_diversities_doo[lognormal_diversities_doo > 0]

pop_diversities = np.zeros(drop_one_pops.shape[0])
for i in range(drop_one_pops.shape[0]):
    pop_diversities[i] = utils.compute_shannon(drop_one_pops[i])

full_pop_shannon = np.zeros(full_pops.shape[0])
for i in range(full_pops.shape[0]):
    full_pop_shannon[i] = utils.compute_shannon(full_pops[i])

glv_diversities = [gaussian_diversities_doo,lognormal_diversities_doo]
tvalues_full = []
mannwhitney_full = []

for i in range(2):
    tvalue = stats.ttest_ind(glv_diversities[i],(pop_diversities),equal_var=False)
    tvalues_full.append(tvalue.pvalue)

    mann_whitney = stats.mannwhitneyu(glv_diversities[i],(pop_diversities),alternative="two-sided")
    mannwhitney_full.append(mann_whitney.pvalue)

print("Mann Whitney p-values: ",mannwhitney_full)
print("T-test p-values: ",tvalues_full)

np.save("../data/figures/fig4/duckweed/gaussian_diversities_full.npy",gaussian_diversities_full)
np.save("../data/figures/fig4/duckweed/lognormal_diversities_full.npy",lognormal_diversities_full)
np.save("../data/figures/fig4/duckweed/gaussian_diversities_doo.npy",gaussian_diversities_doo)
np.save("../data/figures/fig4/duckweed/lognormal_diversities_doo.npy",lognormal_diversities_doo)

difference_lognormal_duckweed = np.zeros(len(lognormal_diversities_doo)*len(pop_diversities))
difference_gaussian_duckweed = np.zeros(len(gaussian_diversities_doo)*len(pop_diversities))

for i in range(len(lognormal_diversities_doo)):
    for j in range(len(pop_diversities)):
        difference_lognormal_duckweed[i*len(pop_diversities)+j] = lognormal_diversities_doo[i] - pop_diversities[j]

for i in range(len(gaussian_diversities_doo)):
    for j in range(len(pop_diversities)):
        difference_gaussian_duckweed[i*len(pop_diversities)+j] = gaussian_diversities_doo[i] - pop_diversities[j]

print(stats.mannwhitneyu(difference_lognormal_duckweed,difference_gaussian_duckweed,alternative="greater"))


#########################
### IN VITRO GUT DATA ###
#########################

data_folder = "../data/interaction_data/ho_et_al/"
full_comm_expt = pd.read_csv(data_folder+"15sp.csv").to_numpy()
full_comm_expt = full_comm_expt[0]
drop_one_out_expt = pd.read_csv(data_folder+"14sp.csv").to_numpy()

R_inf_full = np.loadtxt(data_folder+"R_given.txt",delimiter=",")
Y0_inf_full = np.loadtxt(data_folder+"Y0_given.txt",delimiter=",")

remove_resources = np.array([3,6,14])
if(len(remove_resources)==0):
    Y0_inf = Y0_inf_full
else:
    Y0_inf = np.delete(Y0_inf_full,remove_resources,axis=0)

remove_species = np.array([3,6,14])
if(len(remove_species)==0):
    R_inf = R_inf_full
else:
    R_inf = np.delete(R_inf_full,remove_species,axis=0)
    R_inf = np.delete(R_inf,remove_resources,axis=1)

    Ns,Nr = R_inf.shape
cr_delta = 0.11
cr_supply = Y0_inf
cr_init = np.concatenate((np.full(Ns,0.1),cr_supply))
cr_maxTime = 240

cr_soln = integrate.solve_ivp(competitve_cr, (0,cr_maxTime), cr_init, args=(R_inf,cr_delta,cr_supply,Ns,Nr), method="LSODA")
cr_soln_pops = cr_soln.y[:Ns]

eo_growth,eo_interaction = compute_interactions(cr_soln.y[Ns:,-1],R_inf,cr_supply,cr_delta,Ns,Nr)
rescaled_interactions = np.abs(cr_model.rescale_interactions(eo_interaction, eo_growth))
non_diag_interactions = rescaled_interactions[np.where(rescaled_interactions != 1)]
non_zero_interactions = non_diag_interactions[non_diag_interactions != 0]
sparsity = (1-non_zero_interactions.size/non_diag_interactions.size)

log_mean,log_sd = np.mean(np.log(non_zero_interactions)),np.std(np.log(non_zero_interactions))

plot_interactions = rescaled_interactions.copy()
plot_interactions[np.where(plot_interactions == 0)] = np.nan

plot_r_inf = R_inf_full.copy()
plot_r_inf[np.where(plot_r_inf == 0)] = np.nan

# np.save("../data/figures/fig4/gut/eo_interactions.npy",rescaled_interactions)

simulated_comm_size = 14

init_cond = np.full(simulated_comm_size,0.1)
maxTime = 1000
n_trials = 100

gaussian_diversities = np.zeros(n_trials)
lognormal_diversities = np.zeros(n_trials)

for trialID in range(n_trials):
    gaussian_approx = np.random.normal(np.mean(non_zero_interactions),np.std(non_zero_interactions),size=(simulated_comm_size,simulated_comm_size))
    sparsity_multiplier = np.random.choice([0,1],size=gaussian_approx.shape[0],p=[sparsity,1-sparsity])
    gaussian_approx = gaussian_approx * sparsity_multiplier
    for i in range(gaussian_approx.shape[0]):
        gaussian_approx[i,i] = 1

    lognormal_approx = np.random.lognormal(log_mean,log_sd,size=(simulated_comm_size,simulated_comm_size))
    sparsity_multiplier = np.random.choice([0,1],size=gaussian_approx.shape[0],p=[sparsity,1-sparsity])
    lognormal_approx = lognormal_approx * sparsity_multiplier
    for i in range(lognormal_approx.shape[0]):
        lognormal_approx[i,i] = 1

    gaussian_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(gaussian_approx,), method="LSODA")
    lognormal_soln = integrate.solve_ivp(glv, (0,maxTime), init_cond, args=(lognormal_approx,), method="LSODA")
    
    if((gaussian_soln.y[:,-1]<1e12).all()):
        gaussian_shannon = utils.compute_shannon(gaussian_soln.y[:,-1])
    else:
        gaussian_shannon = -1
    lognormal_shannon = utils.compute_shannon(lognormal_soln.y[:,-1])

    gaussian_diversities[trialID] = gaussian_shannon
    lognormal_diversities[trialID] = lognormal_shannon

gaussian_diversities = gaussian_diversities[gaussian_diversities > 0]
lognormal_diversities = lognormal_diversities[lognormal_diversities > 0]

# np.save("../data/figures/fig4/gut/gaussian_diversities.npy",gaussian_diversities)
# np.save("../data/figures/fig4/gut/lognormal_diversities.npy",lognormal_diversities)

pop_diversities = np.zeros(drop_one_out_expt.shape[0])
for i in range(drop_one_out_expt.shape[0]):
    pop_diversities[i] = utils.compute_shannon(drop_one_out_expt[i])

full_pop_shannon = utils.compute_shannon(full_comm_expt)

glv_diversities = [gaussian_diversities,lognormal_diversities]

print("Drop one out vs GLV")
for i in range(2):
    tvalue = stats.ttest_ind(glv_diversities[i],pop_diversities,equal_var=False)
    print("p-value",tvalue.pvalue)

diversity_difference_lognormal_bhi = np.zeros(len(lognormal_diversities)*len(pop_diversities))
diversity_difference_gaussian_bhi = np.zeros(len(gaussian_diversities)*len(pop_diversities))

for i in range(len(lognormal_diversities)):
    for j in range(len(pop_diversities)):
        diversity_difference_lognormal_bhi[i*len(pop_diversities)+j] = lognormal_diversities[i] - pop_diversities[j]

for i in range(len(gaussian_diversities)):
    for j in range(len(pop_diversities)):
        diversity_difference_gaussian_bhi[i*len(pop_diversities)+j] = gaussian_diversities[i] - pop_diversities[j]
        

print(stats.mannwhitneyu(np.abs(diversity_difference_lognormal_bhi),np.abs(diversity_difference_gaussian_bhi),alternative="less"))