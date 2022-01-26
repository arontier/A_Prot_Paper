#!/usr/bin/env /usr/bin/python
import sys
import numpy as np


def logo():
    print('*********************************************************************')
    print('\
*           _        ____                _   _                      *\n\
*          | |_ _ __|  _ \ ___  ___  ___| |_| |_ __ _               *\n\
*          | __| \'__| |_) / _ \/ __|/ _ \ __| __/ _` |              *\n\
*          | |_| |  |  _ < (_) \__ \  __/ |_| || (_| |              *\n\
*           \__|_|  |_| \_\___/|___/\___|\__|\__\__,_|              *')
    print('*                                                                   *')
    print("* J Yang et al, Improved protein structure prediction using         *\n* predicted interresidue orientations, PNAS, 117: 1496-1503 (2020)  *")
    print("* Please email your comments to: yangjy@nankai.edu.cn               *")
    print('*********************************************************************')


if (len(sys.argv) < 2):

    logo()
    print('\n This script computes the average probability of the top L long+medium-range \n (i.e., |i-j|>=12) predicted contacts from the npz file.\n')
    print(' Please note that higher probability usually yileds more accurate 3D models, \n as indicated in the Figure S3B of the trRosetta paper.\n')

    print(' Example usage: python3 top_prob.py seq.npz\n')
    exit(1)



NPZ = sys.argv[1]
npz = np.load(NPZ)

def top_prob(contacts):
    w = np.sum(contacts['dist'][:,:,1:13], axis=-1)
    L = w.shape[0]
    idx = np.array([[i+1,j+1,0,8,w[i,j]] for i in range(L) for j in range(i+12,L)])
    out = idx[np.flip(np.argsort(idx[:,4]))]
    
    topN=L
    if(out.shape[0]<topN): topN=out.shape[0]
    top=out[0:topN,4].astype(float)
    print("\nAverage probability of the top predicted contacts: %.2f\n" %(np.mean(top)))

logo()
top_prob(npz)

