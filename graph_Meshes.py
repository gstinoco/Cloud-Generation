from scipy.io import loadmat
import os
import glob
import CloudGen

# Variable Initialization.
sizes = ['1', '2', '3']

for me in sizes:
    # Find all the regions.
    regions = glob.glob(f'Meshes/' + me + '/*.mat')
    regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])

    for reg in regions:
        # Load data from the file.
        mat  = loadmat('Meshes/' + me + '/' + reg + '.mat')
        
        mesh = str(me)
        print('Trabajando en la malla ' + reg + '_' + mesh + '.')

        # Get the grid nodes
        x, y  = mat['x'], mat['y']

        # Generate the Cloud
        CloudGen.GraphMesh(x, y, me, reg)