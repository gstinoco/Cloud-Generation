from scipy.io import loadmat
from scipy.io import savemat
import os
import glob
import CloudGen

def run_randomize(Holes):
    # Variable Initialization.
    sizes = ['1', '2', '3']

    for me in sizes:
        # Find all the regions.
        if Holes:
            folder = 'Holes/' + me + '/'
        else:
            folder = 'Clouds/' + me + '/'
        regions = glob.glob(folder + '*.mat')
        regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])

        for reg in regions:
            # Load data from the file.
            mat  = loadmat(folder + reg + '.mat')
            
            mesh = str(me)
            print('Trabajando en la malla ' + reg + '_' + mesh + '.')

            # Get the grid nodes
            p = mat['p']
            tt = mat['tt']

            if Holes:
                folder2 = 'Holes_rand/' + me + '/'
            else:
                folder2 = 'Clouds_rand/' + me + '/'
            
            if not os.path.exists(folder2):
                os.makedirs(folder2)

            nom = folder2 + reg

            # Generate the Cloud
            p = CloudGen.Randomize(p, me)

            CloudGen.GraphCloud(p, save = True, nom = nom, show = False)

            # Save the information of the Cloud.
            nom = nom + '.mat'
            mdic = {"p": p, "tt": tt}
            savemat(nom, mdic)

config = [(True), (False)]

for Holes in config:
    run_randomize(Holes)