import numpy as np
from multiprocessing import Pool, cpu_count
import time
import os

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cr_model
import informed_glv
import utils


def loop_nlist(params):
    START_TIME = time.time()
    Nr,Ns,resSupplyConc,delta,species_params,numRuns,maxTime,seed = params
    np.random.seed(seed)

    supplyVec = np.zeros(Nr)
    supplyVec[:Nr//2] = resSupplyConc 
    results = cr_model.multiple_runs(Nr,Ns,supplyVec,delta,species_params,numRuns,maxTime,seed)
    print(f"Finished run with Nr={Nr}, Ns={Ns} in {time.time()-START_TIME}")
    return results

def multiple_iterator(iterable):
    with Pool(cpu_count()) as p:
        results = p.map(loop_nlist,iterable)
    return results

if __name__ == '__main__':
    print(time.ctime())

    dateStr = time.strftime("%d_%m_%Y_")
    saveTimeStamp = str(int(time.time()))

    # saveFolder = "../../../Data/microbial_interactions/theory/simplified/parameters/"
    saveFolder = "data/"    
    os.mkdir(saveFolder+dateStr+saveTimeStamp+"/")
    dataSaveFolder = saveFolder+dateStr+saveTimeStamp+"/"
   
    run_times = 2
    for runId in range(run_times):

        listlen = 25
        randomSeeds = np.random.randint(0,100000,listlen)

        NsList = np.random.randint(40,200,listlen)
        NrList = np.array([np.random.randint(NsList[i]//2,NsList[i]*2) for i in range(listlen)])

        print(NsList,NrList)

        muMean = np.random.uniform(1,2,listlen)
        muSd = np.random.uniform(1,10,listlen)
        lMin = np.random.uniform(0.1,0.2,listlen)
        lMax = np.random.uniform(0.2,0.6,listlen)
        resSupplyConc = np.full(listlen,50)
        species_params = np.array([muMean,muSd,lMin,lMax]).T

        delta = 0.2
        numRuns = 10
        maxTime = 10000
        comments = "random communities (extra)"

        TSTARTTOTAL = time.time()

        iterable = [[NrList[i],NsList[i],resSupplyConc[i],delta,species_params[i],numRuns,maxTime,randomSeeds[i]] for i in range(len(NsList))]
        functionResults = multiple_iterator(iterable)
        
        print(f"Total time: {time.time()-TSTARTTOTAL}")

        metadata = {"NsList":NsList,"NrList":NrList,"species_params":species_params,"delta":delta,"resSupplyConc":resSupplyConc,"numRuns":numRuns,"maxTime":maxTime,"comments":comments}
        utils.save_compressed_pickle(dataSaveFolder+"scaling_parallel_"+str(runId)+"_"+saveTimeStamp,functionResults)
        utils.save_compressed_pickle(dataSaveFolder+"scaling_parallel_metadata_"+str(runId)+"_"+saveTimeStamp,metadata)

        metadataTxtFile = open(dataSaveFolder+"metdata_"+str(runId)+".txt","w")
        for key in metadata.keys():
            metadataTxtFile.write(f"{key}: {metadata[key]}\n")
        metadataTxtFile.close()

        del functionResults
        del metadata

        print("Finished run",runId)
        
    print(time.ctime())
