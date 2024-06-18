from scipy.io import loadmat
import os
import glob
import csv

# Find all the regions.

regions = glob.glob(f'Clouds_rand/3/*.mat')
regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])
    
for reg in regions:
    # Load data from the file.
    mat  = loadmat('Clouds_rand/3/' + reg + '.mat')

    # Get the cloud nodes
    p, tt  = mat['p'], mat['tt']

    name1 = 'Clouds_rand/3/' + reg + '_p.csv'
    name2 = 'Clouds_rand/3/' + reg + '_tt.csv'

    # Guardar los datos de p en un archivo CSV
    with open(name1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(p)
    
    # Guardar los datos de tt en un archivo CSV
    with open(name2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(tt)