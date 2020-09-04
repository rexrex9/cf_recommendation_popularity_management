from content.datas import filepaths

import sys
import pandas as pd
import random
import math
from tqdm import tqdm




'''
    user_CF
    {
    uid1:{itemid1,itemid2},
    uid2:{itemid1,itemid2}
    }

    item_CF
    {
    itemid1:{uid1,uid2},
    itemid2:{uid1,uid2}
    }
'''
#


def readDatasByPd():
    odatas=pd.read_csv(filepaths.ML_LATEST_SMALL_RATINGS,usecols=[0,1])
    user_dict=dict()
    for d in odatas.values:
        if d[0] in user_dict:
            user_dict[d[0]].add(d[1])
        else:
            user_dict[d[0]] = {d[1]}
    return user_dict

def readItemCfDatasByPd():
    odatas=pd.read_csv(filepaths.ML_LATEST_SMALL_RATINGS,usecols=[0,1])
    dct=dict()
    for d in odatas.values:
        if d[1] in dct:
            dct[d[1]].add(d[0])
        else:
            dct[d[1]] = {d[0]}
    return dct



def getTrainSetAndTestSet(dct):
    trainset,testset = {},{}
    for uid in dct:
        testset[uid] = set(random.sample(dct[uid], math.ceil((0.2 * len(dct[uid])))))
        trainset[uid] = dct[uid]-testset[uid]
    return trainset,testset


def getCosSimRate(s1,s2,trainset,popularities):
    return len(trainset[s1]&trainset[s2])/(len(trainset[s1])*len(trainset[s2]))**0.5

def getIIFSim(s1,s2,trainset,popularities):
    s=0
    for i in trainset[s1]&trainset[s2]:
        s+=1/popularities[i]
    return s/(len(trainset[s1])*len(trainset[s2]))**0.5

def normalizePopularities(popularities):
    maxp=max(popularities.values())
    norm_ppl={}
    for k in popularities:
        norm_ppl[k]=popularities[k]/maxp
    return norm_ppl


def getAlphaSim(s1,s2,trainset,norm_ppl):
    alpha = (1+norm_ppl[s2])/2
    return len(trainset[s1] & trainset[s2]) / (len(trainset[s1])**(1-alpha) * len(trainset[s2])**alpha)

def knn(trainset,k,sim_threshold,popularities,method):
    user_sims={}
    for u1 in tqdm(trainset):
        ulist=[]
        for u2 in trainset:
            if u1!=u2 and len(trainset[u1]&trainset[u2])>0:
                rate = round(method(u1,u2,trainset,popularities),3)
                ulist.append({'id':u2,'rate':rate})
        ulist=[i for i in ulist if i['rate']>sim_threshold]
        user_sims[u1]=sorted(ulist, key=lambda ulist: ulist['rate'], reverse=True)[:k]
    return user_sims

def get_recomedations(user_sims,o_set):
    recomedations = {}
    for u in tqdm(user_sims):
        recomedations[u] = set()
        for sim in user_sims[u]:
            sim_uid = sim['id']
            recomedations[u]|=(o_set[sim_uid] - o_set[u])
    return recomedations

def get_item_CF_recomedations(item_sims,o_set):
    recomedations = {}
    for u in tqdm(o_set):
        recomedations[u] = set()
        for item in o_set[u]:
            for i in item_sims[item]:
                recomedations[u]|=set(j['id'] for j in item_sims[item][:5])-o_set[u]
    return recomedations


def precisionAndRecall(pre,test):
    precision,recall=0,0
    for uid in test:
        t=len(pre[uid]&test[uid])
        precision+=t/(len(pre[uid])+1)
        recall+=t/(len(test[uid])+1)
    return precision/len(test),recall/len(test)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def popularity(x):
    return math.log1p(x)

def getPopularity(item_set):
    p={}
    for k in item_set:
        p[k]=popularity(len(item_set[k]))
    return p

def showPopularity(pre_set,popularities):
    popularity,item_count=0,0
    for uid in pre_set:
        for item in pre_set[uid]:
            popularity+=popularities[item]
            item_count+=1
    return popularity/item_count

def userCF(trainset,popularities,method):
    user_sims = knn(trainset,5,0.01,popularities,method)
    pre_set = get_recomedations(user_sims,trainset)
    return pre_set

def itemCF(trainset,popularities,method,o_set):
    item_sims = knn(trainset,5,0.01,popularities,method)
    pre_set = get_item_CF_recomedations(item_sims,o_set)
    return pre_set


def getEstimate(pre_set,testset,popularities):
    precistion, recall = precisionAndRecall(pre_set, testset)
    popul = showPopularity(pre_set, popularities)
    print('精确率:',precistion)
    print('召回率:',recall)
    print('流行度:',popul)

def play():
    user_trainset,testset = getTrainSetAndTestSet(readDatasByPd())
    items_set = readItemCfDatasByPd()

    itemPopularities = getPopularity(items_set)
    item_norm_ppl=normalizePopularities(itemPopularities)

    userPopularites = getPopularity(user_trainset)
    user_norm_ppl=normalizePopularities(userPopularites)

    print('userCF')
    pre_set_userCF = userCF(user_trainset, itemPopularities, getCosSimRate)
    print('userIIF')
    pre_set_userIIF = userCF(user_trainset, itemPopularities, getIIFSim)
    print('user_alpha')
    pre_set_userAlpha = userCF(user_trainset,user_norm_ppl,getAlphaSim)
    print('itemCF')
    pre_set_itemCF = itemCF(items_set, userPopularites, getCosSimRate,user_trainset)
    print('itemIIFF')
    pre_set_itemIIF = itemCF(items_set, userPopularites, getIIFSim, user_trainset)
    print('item_alpha')
    pre_set_itemAlpha = itemCF(items_set, item_norm_ppl, getAlphaSim,user_trainset)

    print('userCF')
    getEstimate(pre_set_userCF,testset,itemPopularities)
    #
    print('userIIF')
    getEstimate(pre_set_userIIF,testset,itemPopularities)

    print('userAlpha')
    getEstimate(pre_set_userAlpha,testset,itemPopularities)

    print('itemCF')
    getEstimate(pre_set_itemCF,testset,itemPopularities)

    print('itemIIFF')
    getEstimate(pre_set_itemIIF,testset,itemPopularities)

    print('ItemAlpha')
    getEstimate(pre_set_itemAlpha,testset,itemPopularities)

if __name__ == '__main__':
    play()




'''
userCF
精确率: 0.0523414093788954
召回率: 0.5807093085093415
流行度: 3.1754908371847206
userIIF
精确率: 0.04591475090899522
召回率: 0.6273353066587025
流行度: 2.9678558215127344
userAlpha
精确率: 0.09197783794303584
召回率: 0.37284283245655453
流行度: 3.8165634839891758
itemCF
精确率: 0.07626677973686181
召回率: 0.5381131908798534
流行度: 3.025877220922286
itemIIFF
精确率: 0.08958159294184373
召回率: 0.5878472322920266
流行度: 3.0224481726545225
ItemAlpha
精确率: 0.03809527281425079
召回率: 0.38687794423699895
流行度: 2.709847691028236

'''

