#!/usr/bin/env python
import numpy as np
import sys,os,getopt
import scipy
import scipy.spatial
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





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



DICT = {
    'ALA':'A',
    'ARG':'R',
    'ASN':'N',
    'ASP':'D',
    'CYS':'C',
    'GLN':'Q',
    'GLU':'E',
    'GLY':'G',
    'HIS':'H',
    'ILE':'I',
    'LEU':'L',
    'LYS':'K',
    'MET':'M',
    'PHE':'F',
    'PRO':'P',
    'SER':'S',
    'THR':'T',
    'TRP':'W',
    'TYR':'Y',
    'VAL':'V',
    
    'ASX':'N',
    'GLX':'Q',
    'UNK':'G',
    'HSD':'H',
    }



def get_dihedrals(a, b, c, d):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)

    return np.arctan2(y, x)

def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v * w, axis=1)

    return np.arccos(x)

def parse_pdb(pdbfile,chain_number):
    dict_pdb=defaultdict(list)
    fr=open(pdbfile)
    lines=fr.readlines()
    for line in lines:
        if line.startswith('ATOM'):
            chain=line[21]
            dict_pdb[chain].append(line)
    chains=list(dict_pdb.keys())
    key=chains[chain_number-1]
    return (dict_pdb[key])

def get_xyz(pdb):
    target=['N','CA','CB','C']
    dict_xyz_all=defaultdict(dict)
    for line in pdb:
        residue = line[17:20].strip()
        ####remove nonstandard amino acid
        if DICT.get(residue):
            atomname = line[12:16].strip()
            if atomname in target:
                res_no=line[22:26].strip()
                coords_str=[line[30:38], line[38:46], line[46:54]]
                coords=[float(k) for k in coords_str]
                dict_xyz_all[res_no][atomname]=coords
    essential=['N','CA','C']
    dict_xyz={}
    for key in dict_xyz_all:
        item=dict_xyz_all[key]
        atom_set=item.keys()
        if set(essential).issubset(set(atom_set)):
            dict_xyz[key]=item


    return (dict_xyz)


def get_info(xyz,dmax):
    nres= len(xyz)
    position=list(xyz.keys())
    N = np.stack([xyz[position[i]]['N'] for i in range(nres)])
    CA = np.stack([xyz[position[i]]['CA'] for i in range(nres)])
    C= np.stack([xyz[position[i]]['C'] for i in range(nres)])
    # recreate Cb given N,Ca,C
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
    ###replace CB with real xyz
    for i in range(nres):
        value=xyz[position[i]]
        if 'CB' in value.keys():
             CB[i]=xyz[position[i]]['CB']

    # fast neighbors search
    kdCb = scipy.spatial.cKDTree(CB)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0, idx1] = np.linalg.norm(CB[idx1] - CB[idx0], axis=-1)
    ##Cb-Cb distance matrix for plot
    dist6d_plot = np.eye(nres)*(-16)+20
    dist6d_plot[idx0, idx1] = np.linalg.norm(CB[idx1] - CB[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0, idx1] = get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1])
    #matrix of Ca-Cb-Cb-Ca dihedrals for plot
    omega6d_plot = np.eye(nres)*(-180)+180
    omega6d_plot[idx0, idx1] = np.rad2deg(get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1]))

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0, idx1] = get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1])
    # matrix of polar coord theta for plot
    theta6d_plot = np.eye(nres)*(-180)+180
    theta6d_plot[idx0, idx1] = np.rad2deg(get_dihedrals(N[idx0], CA[idx0], CB[idx0], CB[idx1]))

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0, idx1] = get_angles(CA[idx0], CB[idx0], CB[idx1])
    # matrix of polar coord phi for plot
    phi6d_plot = np.eye(nres)*(-180)+180
    phi6d_plot[idx0, idx1] = np.rad2deg(get_angles(CA[idx0], CB[idx0], CB[idx1]))
    
    return (dist6d, dist6d_plot, omega6d, omega6d_plot,theta6d, theta6d_plot, phi6d, phi6d_plot)
    

def plot(name,img):
    plt.figure()
    color_map = plt.imshow(img.astype(float))
    color_map.set_cmap("hot")
    plt.colorbar()
    plt.savefig(name, bbox_inches='tight', dpi=600, transparent=False)



def output(path,xyz):
    (dist6d, dist6d_plot, omega6d, omega6d_plot,theta6d, theta6d_plot, phi6d, phi6d_plot)=get_info(xyz,20)
    ###generate npz file
    npzfile=path+'.npz'
    np.savez_compressed(npzfile,dist=dist6d,omega=omega6d,theta=theta6d,phi=phi6d)
    ###plot distance map
    name_dist=path+'.'+'dist'+'.png'
    plot(name_dist,dist6d_plot)
    ###plot omega map
    name_omega=path+'.'+'omega'+'.png'
    plot(name_omega,omega6d_plot)
    ###plot theta map
    name_theta = path + '.' + 'theta' + '.png'
    plot(name_theta, theta6d_plot)
    ###plot phi map
    name_phi = path + '.' + 'phi' + '.png'
    plot(name_phi, phi6d_plot)

def usage():
    print(" usage: python3 pdb2npz.py [options]")
    print(" options:")
    print(" -f pdb file (mandatory)")
    print(" -n chain number (optional), default: 1 (i.e., the first chain in pdb file)")

def main():
    opts, args = getopt.getopt(sys.argv[1:], "f:n:h")
    pdbfile='-1'
    chain_number = 1  ##default the first chain
    for o, a in opts:
        if o == "-f":
            pdbfile = a
        elif o == "-n":
            chain_number=int(a)
        elif o == '-h':
            usage()
            sys.exit(0)

    if (pdbfile == '-1'):
        logo()
        print('\n This script calculate and visualize the interresidue geometries (distance and orientation)\n from the input of a PDB structure.\n')
        usage()
        exit(1)

    logo()

    abspath = os.path.abspath(pdbfile)
    fname=os.path.basename(abspath).split('.')[0]
    datadir = os.path.dirname(abspath)
    pdb=parse_pdb(pdbfile,chain_number)
    xyz=get_xyz(pdb)
    path=datadir+'/'+fname
    print("computing distance and orientation...")
    output(path,xyz)
    print("the interresidue geometries are saved in ", path+'.npz')
    print("the visualization is in *.png")
    print("done")
   
    

if __name__ == "__main__":
    main()

