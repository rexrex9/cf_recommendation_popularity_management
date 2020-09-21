__author__='雷克斯掷骰子'
'''
B站:https://space.bilibili.com/497998686
头条:https://www.toutiao.com/c/user/token/MS4wLjABAAAAAxu8A9lNX1qfkRKEyU9Uecqa2opPcZufDLWHbv7m-hVdMVPOe7r_i-k6nw4RY61i/
'''
'''
数据下载地址：https://grouplens.org/datasets/movielens/
'''
import pandas as pd
import random
import math
from tqdm import tqdm
import sys

#pip install tqdm

def readDatas():
    path = 'ml-latest-small/ratings.csv'
    odatas = pd.read_csv(path,usecols=[0,1])
    user_dict=dict()
    for d in odatas.values:
        if d[0] in user_dict:
            user_dict[d[0]].add(d[1])
        else:
            user_dict[d[0]] = {d[1]}
    return user_dict

def readItemDatas():
    path = 'ml-latest-small/ratings.csv'
    odatas = pd.read_csv(path, usecols=[0, 1])
    item_dict=dict()
    for d in odatas.values:
        if d[1] in item_dict:
            item_dict[d[1]].add(d[0])
        else:
            item_dict[d[1]] = {d[0]}
    return item_dict


def getTrainsetAndTestset(dct):
    trainset,testset=dict(),dict()
    for uid in dct:
        testset[uid]=set(random.sample(dct[uid],math.ceil(0.2*len(dct[uid]))))
        trainset[uid] = dct[uid]-testset[uid]
    return trainset,testset

def cossim(s1,s2,trainset,popularities):
    return len(trainset[s1]&trainset[s2])/(len(trainset[s1])*len(trainset[s2]))**0.5


def iifsim(s1,s2,trainset,popularities):
    s=0
    for i in trainset[s1] & trainset[s2]:
        s+=1/popularities[i]
    return s/ (len(trainset[s1]) * len(trainset[s2])) ** 0.5

def normalizePolularities(polularities):
    maxp = max(polularities.values())
    norm_ppl = {}
    for k in polularities:
        norm_ppl[k] = polularities[k]/maxp
    return norm_ppl

def alphaSim(s1,s2,trainset,nolms):
    alpha = (1+nolms[s2])/2
    return len(trainset[s1]&trainset[s2])/(len(trainset[s1])**(1-alpha)*len(trainset[s2])**alpha)

def knn(trainset,k,method,popularities):
    user_sims={}
    for u1 in tqdm(trainset):
        ulist=[]
        for u2 in trainset:
            if u1==u2 or len(trainset[u1]&trainset[u2])==0: continue
            rate = method(u1,u2,trainset,popularities)
            ulist.append({'id':u2,'rate':rate})
        user_sims[u1]=sorted(ulist,key=lambda ulist:ulist['rate'],reverse=True)[:k]
    return user_sims

def get_recomedations(user_sims,o_set):
    recomedation = dict()
    for u in tqdm(user_sims):
        recomedation[u] = set()
        for sim in user_sims[u]:
            recomedation[u] |= (o_set[sim['id']] - o_set[u])
    return recomedation

def get_recomedations_by_itemCF(item_sims,o_set):
    recomedation = dict()
    for u in tqdm(o_set):
        recomedation[u] = set()
        for item in o_set[u]:
            recomedation[u] |= set(i['id'] for i in item_sims[item])-o_set[u]
    return recomedation

def precisionAndRecall(pre,test):
    p,r = 0,0
    for uid in test:
        t=len(pre[uid]&test[uid])
        p+=t/(len(pre[uid])+1)
        r+=t/(len(test[uid])+1)
    return p/len(test),r/len(test)

def popularity(x):
    return math.log1p(x)
    #return x

def getPopularity(item_sets):
    p=dict()
    for item_id in item_sets:
        p[item_id]=popularity(len(item_sets[item_id]))
    return p

def totalPopularity(pre_set,popularities):
    p,item_count=0,0
    for uid in pre_set:
        for item in pre_set[uid]:
            p+=popularities[item]
            item_count+=1
    return p/item_count

def getEstimate(pre_set,testset,popularities):
    precision,recall=precisionAndRecall(pre_set, testset)
    popul=totalPopularity(pre_set,popularities)
    print('精确率：',precision)
    print('召回率: ',recall)
    print('流行度: ',popul)


def play():
    odatas = readDatas()
    item_datas = readItemDatas()
    trset,teset = getTrainsetAndTestset(odatas)

    itemPopularities = getPopularity(item_datas)
    userPopularities = getPopularity(trset)

    itemPopularities_norm = normalizePolularities(itemPopularities)
    userPopularities_norm = normalizePolularities(userPopularities)

    user_sims = knn(trset,5,cossim,itemPopularities)
    user_iff_sims = knn(trset,5,iifsim,itemPopularities)
    user_alpha_sims = knn(trset,5,alphaSim,userPopularities_norm)
    item_sims = knn(item_datas, 5,cossim,userPopularities)
    item_iif_sims=knn(item_datas,5,iifsim,userPopularities)
    item_alpha_sims = knn(item_datas,5,alphaSim,itemPopularities_norm)


    pre_set = get_recomedations(user_sims,trset)
    pre_set_iif = get_recomedations(user_iff_sims,trset)
    pre_set_user_alpha = get_recomedations(user_alpha_sims,trset)

    pre_itemCF_set = get_recomedations_by_itemCF(item_sims,trset)
    pre_itemiif_set = get_recomedations_by_itemCF(item_iif_sims,trset)
    pre_set_item_alpha = get_recomedations_by_itemCF(item_alpha_sims, trset)

    print('userCF')
    getEstimate(pre_set,teset,itemPopularities)
    print('userIIF')
    getEstimate(pre_set_iif,teset,itemPopularities)
    print('userAlpha')
    getEstimate(pre_set_user_alpha,teset,itemPopularities)

    print('itemCF')
    getEstimate(pre_itemCF_set, teset, itemPopularities)
    print('itemIIF')
    getEstimate(pre_itemiif_set, teset, itemPopularities)
    print('itemAlpha')
    getEstimate(pre_set_item_alpha, teset,itemPopularities)

if __name__ == '__main__':
    play()
