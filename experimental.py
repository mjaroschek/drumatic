# Notebook for development

import os
os.chdir("/home/maximilian/drummidi/")

from Pipeline import Pipeline
from Signal import *
from drum_environment import*
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

def max_up(s,k):
    b=int(np.ceil(len(s[0])/k))
    newS=[[0 for j in i] for i in s]
    for i in range(len(s)):
        for j in range(b):
            newS[i][k*j:k*(j+1)]=[max([access(s,i,k*j+l) for l in range(k)])]*k
    return newS

    # newS=[[0 for j in i] for i in s]
    # for i in range(len(s)):
    #     for j in range(len(s[0])):
    #         newS[i][j]=max([access(s,i+l,j) for l in range(k)])
    # return newS

def move_up(s):
    m=abs(min([min([j for j in i if np.isnan(j)==False]) for i in s]))
    return [[i+m if np.isnan(i)==False else 0 for i in j] for j in s]
    
def sum_up(s,k):
    b=int(np.ceil(len(s[0])/k))
    newS=[[0 for j in i] for i in s]
    for i in range(len(s)):
        for j in range(b):
            newS[i][k*j:k*(j+1)]=[sum([access(s,i,k*j+l) for l in range(k)])]*k
    return newS
    
def transpose(s):
    return [[i[j] for i in s] for j in range(len(s[0]))]

def access(s,i,j=None):
    if j==None:
        if type(s[0])=="list":
            if i<0 or i>=len(s):
                return []
            else:
                return s[i]
        else:
            if i<0 or i>=len(s):
                return 0
            else:
                return s[i]
    if i<0 or i>=len(s) or j<0 or j>=len(s[0]):
        return 0
    return s[i][j]

def convolve(m,k):
    m=transpose(m)
    new_m=[[0 for j in i] for i in m]
    c_i=int(np.floor(len(k)/2))
    c_j=int(np.floor(len(k[0])/2))
    for i in range(len(m)):
        for j in range(len(m[0])):
            v=0
            for k_i in range(len(k)-1,-1,-1):
                for k_j in range(len(k[0])-1,-1,-1):
                    v=v+k[k_i][k_j]*access(m,len(k)-k_i-1+i-c_i,len(k[0])-k_j-1+j+-c_j)
            new_m[i][j]=v
    return transpose(new_m)
    
def normalize(s):
    m=max(sum(s,[]))
    return [[j/m for j in i] for i in s]

def deep_map(l,f,it=1):
    for k in range(it):
        l=[[f(j) for j in i] for i in l]
    return l

# def refine(spec):
#     new=[[0 for i in j] for j in spec]
#     sleep_timer=10
#     sleep=0
#     for i,s in enumerate(spec):
#         a=np.median(s)
#         ratio=0.0
#         b=(ratio+a*(1-ratio))

#         for j,t in enumerate(s):
#             a=[abs(access(s,j-k)) for k in range(2,6)]
#             a.sort()
#             a=a[1]
#             b=(ratio+a*(1-ratio))
#             if t>=b:
#                 new[i][j]=1
#     new=transpose(new)
#     for i,s in enumerate(new):
#         if sum(s)>0.8*len(s) and sleep==0:
#             new[i]=[1 for j in s]
#             sleep=sleep_timer+1
#         else:
#             new[i]=[0 for j in s]
#         if sleep!=0:
#             sleep=sleep-1
#     return transpose(new)

def refine(spec):
    new=[[0 for i in j] for j in spec]
    sleep_timer=10
    sleep=0
    for i,s in enumerate(spec):
        a=np.median(s)
        ratio=0.0
        b=(ratio+a*(1-ratio))

        for j,t in enumerate(s):
            a=[abs(access(s,j-k)) for k in range(2,5)]
            a.sort()
            a=a[0]
            b=(ratio+a*(1-ratio))
            if t>=b:
                new[i][j]=1
    new=transpose(new)
    onsets=0
    for i,s in enumerate(new):
        counter=0
        found=0
        for j,p in enumerate(s):
            if p==1 or (access(new,i-1,j)==1 or access(new,i+1,j)==1):
                counter=counter+1
            else:
                # if counter>=3:
                #     found=found+1
                if counter>=7:
                    found=found+4
                counter=0
        if found>3 and sleep==0:
            onsets=onsets+1
            new[i]=[1 for j in s]
            sleep=sleep_timer+1
        else:
            new[i]=[0 for j in s]
        if sleep!=0:
            sleep=sleep-1
    return transpose(new)

