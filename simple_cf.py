

'''
数据下载地址：https://grouplens.org/datasets/movielens/
'''
import pandas as pd
import random
import math
from tqdm import tqdm

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

def cossim(s1,s2,trainset):
    return len(trainset[s1]&trainset[s2])/(len(trainset[s1])*len(trainset[s2]))**0.5

def knn(trainset,k):
    user_sims={}
    for u1 in tqdm(trainset):
        ulist=[]
        for u2 in trainset:
            if u1==u2 or len(trainset[u1]&trainset[u2])==0: continue
            rate = cossim(u1,u2,trainset)
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

def play():
    odatas = readDatas()
    item_datas = readItemDatas()
    trset,teset = getTrainsetAndTestset(odatas)

    user_sims = knn(trset,5)
    item_sims = knn(item_datas, 5)
    
    pre_set = get_recomedations(user_sims,trset)
    pre_itemCF_set = get_recomedations_by_itemCF(item_sims,trset)

    p,r = precisionAndRecall(pre_set,teset)
    pi,ri = precisionAndRecall(pre_itemCF_set,teset)

    print('userCF')
    print(p,r)
    print('itemCF')
    print(pi,ri)

if __name__ == '__main__':
    play()


