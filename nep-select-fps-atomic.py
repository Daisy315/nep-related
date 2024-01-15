from pynep.calculate import NEP
from pynep.select import FarthestPointSample
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multiprocessing import Pool
import sys

if __name__=='__main__':
    # read old training data, new data and descriptor
    fxyz = "/home/fangmd/workdir/0-paper-2/2-FPS/1-pre_cutoff-4/PCA"
    data_ref=read(fxyz+"/c3.xyz",index=':',format='extxyz')
    data_current=read(fxyz+"/c4.xyz",index=':',format='extxyz')
    calc = NEP(fxyz+"/nep.txt")
    
    lat_ref = np.concatenate([calc.get_property('descriptor', atoms) for atoms in data_ref])
    lat_current = np.concatenate([calc.get_property('descriptor', atoms) for atoms in data_current])
    comesfrom = np.concatenate([[i] * len(atoms) for i, atoms in enumerate(data_current)])
    sampler = FarthestPointSample()
    indices = [comesfrom[i] for i in sampler.select(lat_current, lat_ref, min_select=100)]
    indices = set(indices)
    write('selected.xyz', [data_current[i] for  i in indices],format='extxyz')
    with open('selected_i','w') as f:
        f.write(str(indices))
    
    reducer = PCA(n_components=2)
    reducer.fit(lat_current)
    
    # current data
    proj_current = reducer.transform(lat_current)
    plt.scatter(proj_current[:,0], proj_current[:,1],label="init data",color="orange")
    # reference data
    proj_ref=reducer.transform(lat_ref)
    plt.scatter(proj_ref[:,0],proj_ref[:,1], label="init data",s=1,color="silver")
    # selected data
    proj_selected = reducer.transform(np.array([lat_current[i] for i in indices]))
   
    np.savetxt("proj_ref",proj_ref)
    np.savetxt("proj_current",proj_current)
    np.savetxt("proj_selected",proj_selected)

    plt.legend()
    plt.axis('off')
    plt.savefig('select.png')