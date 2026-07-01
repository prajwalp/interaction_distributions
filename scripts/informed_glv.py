import numpy as np
import sys 
import utils

from numba import njit,cfunc,carray
from numbalsoda import lsoda_sig,lsoda

def compute_correlated(rescaled_inter,Ns):
    frac_pp = []
    frac_pp_1 = []
    frac_pp_2 = []
    frac_nn = []
    frac_nn_1 = []
    frac_nn_2 = []

    for i in range(Ns):
        for j in range(i):
                ij = rescaled_inter[i,j]
                ji = rescaled_inter[j,i]
                if(ij>0 and ji>0):
                    frac_nn.append(ij)
                    frac_nn.append(ji)
                    frac_nn_1.append(ij)
                    frac_nn_2.append(ji)
                elif(ij<0 and ji<0):
                    frac_pp.append(ij)
                    frac_pp.append(ji)
                    frac_pp_1.append(ij)
                    frac_pp_2.append(ji)

    frac_nn = np.array(frac_nn)
    frac_pp = np.array(frac_pp)
    frac_pp_1 = np.array(frac_pp_1)
    frac_pp_2 = np.array(frac_pp_2)
    frac_nn_1 = np.array(frac_nn_1)
    frac_nn_2 = np.array(frac_nn_2)
    return frac_nn,frac_pp,frac_pp_1,frac_pp_2, frac_nn_1,frac_nn_2


def compute_correlation_fit_params(rescaled_inter,Ns):
    frac_nn,frac_pp,frac_pp_1,frac_pp_2,frac_nn_1,frac_nn_2= compute_correlated(rescaled_inter,Ns)

    if(frac_pp.size>0):
        pp_fit = np.array([np.mean(np.log(-frac_pp)),np.std(np.log(-frac_pp))])
        cov = (np.mean((np.log(-frac_pp_1)-np.mean(np.log(-frac_pp_1)))*(np.log(-frac_pp_2)-np.mean(np.log(-frac_pp_2)))))
    else:
        pp_fit = np.array([np.nan,np.nan])
        cov = np.nan
    nn_fit = np.array([np.mean(np.log(frac_nn)),np.std(np.log(frac_nn))])
    cov_nn =  (np.mean((np.log(frac_nn_1)-np.mean(np.log(frac_nn_1)))*(np.log(frac_nn_2)-np.mean(np.log(frac_nn_2)))))

    return np.concatenate([[frac_nn.size,frac_pp.size],pp_fit,nn_fit,[cov,cov_nn]])


def compute_lg_fit_params(rescaled_inter,Ns):
    non_diag_inters = rescaled_inter[rescaled_inter!=1]
    pos_inters,neg_inters = -non_diag_inters[non_diag_inters<0],non_diag_inters[non_diag_inters>0]
    frac_neg = neg_inters.size/(pos_inters.size+neg_inters.size)
    if(frac_neg<1):
        pos_mean,pos_sd = np.mean(np.log(pos_inters)),np.std(np.log(pos_inters))
    else:
        pos_mean,pos_sd = np.nan,np.nan
    neg_mean,neg_sd = np.mean(np.log(neg_inters)),np.std(np.log(neg_inters))
    return frac_neg,pos_mean,pos_sd,neg_mean,neg_sd


def generate_correlated_matrix(corr_fits,Ns):
    test_inter_matrix = np.zeros((Ns,Ns))

    size_nn,size_pp,cov,cov_nn = corr_fits[0], corr_fits[1], corr_fits[-2], corr_fits[-1]
    pp_fit = corr_fits[2:4]
    nn_fit = corr_fits[4:6]
    if(np.isnan(pp_fit[0])):
        size_nn = Ns*(Ns-1)
        size_pp = 0
    cov_matrix = np.array([[pp_fit[1]**2, cov], [cov, pp_fit[1]**2]])
    cov_matrix_nn = np.array([[nn_fit[1]**2, cov_nn], [cov_nn, nn_fit[1]**2]])

    for i in range(Ns):
        for j in range(i):
            rand_total = Ns*(Ns-1)
            random_check = np.random.random()
            if(random_check<=size_nn/rand_total):
                correlated_normal = np.random.multivariate_normal(np.ones(2)*nn_fit[0],cov_matrix_nn)
                test_inter_matrix[i,j] = np.exp(correlated_normal[0])
                test_inter_matrix[j,i] = np.exp(correlated_normal[1])
            elif(random_check<(size_nn+size_pp)/rand_total):
                correlated_normal = np.random.multivariate_normal(np.ones(2)*pp_fit[0],cov_matrix)
                test_inter_matrix[i,j] = -np.exp(correlated_normal[0])
                test_inter_matrix[j,i] = -np.exp(correlated_normal[1])
            else:
                random_check2 = np.random.random()

                if(random_check2<=0.5):
                    test_inter_matrix[i,j] = -np.random.lognormal(pp_fit[0],pp_fit[1])
                    test_inter_matrix[j,i] = np.random.lognormal(nn_fit[0],nn_fit[1])
                else:
                    test_inter_matrix[i,j] = np.random.lognormal(nn_fit[0],nn_fit[1])
                    test_inter_matrix[j,i] = -np.random.lognormal(pp_fit[0],pp_fit[1])
        test_inter_matrix[i,i] = 1

    return test_inter_matrix

def generate_lg_matrix(lg_fits,Ns):
    test_inter_matrix2 = np.zeros((Ns,Ns))
    frac_neg,pos_mean,pos_sd,neg_mean,neg_sd = lg_fits
    if(pos_sd<0 and frac_neg<1):
        print("Warning: Negative SD for positive interactions")
    for i in range(Ns):
        for j in range(Ns):
            if(i==j):
                test_inter_matrix2[i,j] = 1
            else:
                rand_check = np.random.random()
                if(rand_check<frac_neg or pos_sd<0):
                    test_inter_matrix2[i,j] = np.random.lognormal(neg_mean,neg_sd)
                else:
                    test_inter_matrix2[i,j] = -np.random.lognormal(pos_mean,pos_sd)

    return test_inter_matrix2

def generate_gaussian_matrix(gauss_fit,Ns):
    gaussian_approx = np.random.normal(gauss_fit[0],gauss_fit[1],(Ns,Ns))
    for i in range(Ns):
        gaussian_approx[i,i] = 1

    return gaussian_approx


def make_lsoda_func_glv_dynamics(alpha, Ns):
    @cfunc(lsoda_sig)
    def glv_dynamics_lsoda_log(t, logx, dx, p):
        x_ = carray(logx, (Ns,)); dx_ = carray(dx, (Ns,))
        y = x_[:Ns]
        ydot = 1 - np.dot(alpha, np.exp(y))
        ydot[y <= -30] = 0
        ydot[y >= 20] = 0
        dx_[:Ns] = ydot
    return glv_dynamics_lsoda_log

def simulate_stability(interaction,maxTime,Ns):
    init_cond = np.full(Ns,0.1)
    tGLV = np.linspace(0, maxTime, 1000)
    glv_dynamics_lsoda = make_lsoda_func_glv_dynamics(interaction,Ns)
    funcptr_glv = glv_dynamics_lsoda.address
    glv_soln,success = lsoda(funcptr_glv, np.log(init_cond), tGLV)            

    if((glv_soln[-1,:]<20).all()):
        glv_pop = np.exp(glv_soln[-1,:])
        return_vals =  1,np.where(glv_pop>1e-3)[0].size,utils.compute_shannon(glv_pop)
        del glv_soln
        return return_vals
    else:
        return 0,np.nan,np.nan