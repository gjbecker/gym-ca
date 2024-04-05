import h5py
import os

filename = '/work/flemingc/gjbecker/gym_ca/GA3C/4_agent_29_actions/dataset.hdf5'
filename = '/work/flemingc/gjbecker/gym_ca/GA3C/4_agent_11_actions/dataset.hdf5'

file = h5py.File(filename, 'r')

print(file.keys())
for k in file.keys():
    if k != 'metadata':
        print(k, file[k][()].shape)

print(file['metadata']['source'][()])
print(file['metadata']['time'][()])
file.close()
