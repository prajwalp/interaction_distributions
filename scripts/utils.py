import numpy as np
import matplotlib.pyplot as plt

from numba import njit
import pickle
import bz2
import _pickle as cPickle

########################
### HELPER FUNCTIONS ###
########################

def straight_line(x,m,c):
    return m*x + c
  
def save_compressed_pickle(filename, data):
    with bz2.BZ2File(filename + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)

def load_compressed_pickle(filename):    
    data = bz2.BZ2File(filename + '.pbz2', 'rb')
    data = cPickle.load(data)
    return data

def normal_to_lognormal(mean,sd):
    mu = np.round(np.log(mean**2/np.sqrt(mean**2 + sd**2)),8)
    sigma = np.round(np.sqrt(np.log(1 + sd**2/mean**2)),8)
    return mu,sigma

def generate_axis(minVal,maxVal,nBins,logFlag):
    if(logFlag):
        return np.geomspace(minVal,maxVal,nBins)
    else:
        return np.linspace(minVal,maxVal,nBins)

##########################
###  COMPUTE FUNCTIONS ###
##########################

def compute_survived_indices(chemostatPopData,Ns,numRuns,threshold=0.1):
    survivedIndicesList = []
    for i in range(numRuns):
        survivedIndices = np.where(chemostatPopData[i][:Ns] > threshold)[0]
        survivedIndicesList.append(survivedIndices)
    return survivedIndicesList

def survived_across_params(data,metadata,threshold,acrossNsFlag = False):
    NsList,numRuns = metadata["NsList"],metadata["numRuns"]
    paramListLen = len(data)
    survivedList,numSurvivedSpecies = [],[]
    
    for i in range(paramListLen):
        if(acrossNsFlag):
            Ns = NsList[i]
        else:
            Ns = NsList[0]
        chemostatPopData = data[i][7]
        survivedIndicesList = compute_survived_indices(chemostatPopData,Ns,numRuns,threshold)
        survivedList.append(survivedIndicesList)
        numSurvivedSpecies.append([len(survivedIndices) for survivedIndices in survivedIndicesList])
    return survivedList,numSurvivedSpecies

def generate_full_interaction_statistics(data,metadata,acrossNsFlag = False):
    NsList,numRuns = metadata["NsList"],metadata["numRuns"]
    paramListLen = len(data)
    cumulants,err_cumulants = np.zeros((3,paramListLen)),np.zeros((3,paramListLen))
                                                                                
    for i in range(paramListLen):
        analyzedInteractions = data[i][6]*(np.ones(data[i][6].shape) - np.eye(data[i][6].shape[-1]))
        factor = NsList[i]/(NsList[i] - 1)
        assert (np.diag(analyzedInteractions[0]) == 0).all()
        meanWithinSample,varWithinSamples = np.mean(analyzedInteractions,axis=(1,2)),np.var(analyzedInteractions,axis=(1,2))
        thirdCWithinSample = factor*np.mean((analyzedInteractions - meanWithinSample[:,np.newaxis,np.newaxis])**3,axis=(1,2))
        withinSampleStastics = [meanWithinSample,varWithinSamples,thirdCWithinSample]

        for j in range(3):
            cumulants[j,i] = np.mean(withinSampleStastics[j])
            err_cumulants[j,i] = np.std(withinSampleStastics[j])/np.sqrt(numRuns)
    return cumulants,err_cumulants

def generate_survived_interaction_statistics(data,metadata,threshold=0.1,acrossNsFlag = False):
    survivedMean,survivedVar,survivedThirdC = [],[],[]
    fracSurvMean,fracSurvVar,fracSurvThirdC = [],[],[]
    NsList,numRuns = metadata["NsList"],metadata["numRuns"]
    paramListLen = len(data)

    for i in range(paramListLen):
        if(acrossNsFlag):
            Ns = NsList[i]
        else:
            Ns = NsList[0]
        for j in range(numRuns):
            chemostatPopData,rescaledInteractions = data[i][7][j],data[i][6][j]
            survivedIndices = np.where(chemostatPopData[:Ns] > threshold)[0]
            survNs = len(survivedIndices)
            survivedEffInteractions = (rescaledInteractions[survivedIndices][:,survivedIndices])
            assert (np.diag(survivedEffInteractions)==1).all()
            survivedEffInteractions = survivedEffInteractions - np.diag(np.diag(survivedEffInteractions))
            if(survNs > 1):
                factor = survNs/(survNs - 1)
            else:
                factor = 1

            meanSurvivedEffInteractions = np.mean(survivedEffInteractions)*factor
            varSurvivedEffInteractions = np.var(survivedEffInteractions)*factor
            thirdCSurvivedEffInteractions = np.mean((survivedEffInteractions - meanSurvivedEffInteractions)**3)*factor
            survivedMean.append([survNs,meanSurvivedEffInteractions])
            survivedVar.append([survNs,varSurvivedEffInteractions])
            survivedThirdC.append([survNs,thirdCSurvivedEffInteractions])
            fracSurvMean.append([survNs/Ns,meanSurvivedEffInteractions])
            fracSurvVar.append([survNs/Ns,varSurvivedEffInteractions])
            fracSurvThirdC.append([survNs/Ns,thirdCSurvivedEffInteractions])

    argSorting = np.argsort(np.array(survivedMean)[:,0])
    survivedMean,survivedVar,survivedThirdC = np.array(survivedMean),np.array(survivedVar),np.array(survivedThirdC)
    survivedCumulants = [survivedMean[argSorting],survivedVar[argSorting],survivedThirdC[argSorting]]

    argSorting = np.argsort(np.array(fracSurvMean)[:,0])
    fracSurvMean,fracSurvVar,fracSurvThirdC = np.array(fracSurvMean),np.array(fracSurvVar),np.array(fracSurvThirdC)
    fracSurvCumulants = [fracSurvMean[argSorting],fracSurvVar[argSorting],fracSurvThirdC[argSorting]]
    
    return survivedCumulants,fracSurvCumulants
