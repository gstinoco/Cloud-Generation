import os
import glob
import CloudGen
import pandas as pd

def run(Holes):
    # Variable Initialization.
    sizes = [1, 2, 3]
    first = True

    for me in sizes:
        size = str(me)
        if first:
            # Find all the regions.
            regions = glob.glob(f'Contours/*.csv')
            regions = sorted([os.path.splitext(os.path.basename(region))[0] for region in regions])
            first = False
    
        for reg in regions:
            print(f'Working on {reg}_{size}.')

            # Load data from the file.
            df = pd.read_csv('Contours/' + reg + '.csv', header = None, names = ['x', 'y'])

            # Get the boundary nodes
            x = df['x'].values.reshape(-1, 1)
            y = df['y'].values.reshape(-1, 1)

            # Generate the Cloud
            if Holes:
                if reg == 'ENG':
                    h_coords = [(0.3, 0.3, 0.05)]
                elif reg == 'PAT':
                    h_coords = [(0.25, 0.32, 0.05)]
                else:
                    h_coords = [(x.max()/2, y.max()/2, 0.05)]
                p, tt = CloudGen.CreateCloud_Holes(x, y, (2/3)*me, h_coor = h_coords)
            else:
                p, tt = CloudGen.CreateCloud(x, y, (2/3)*me) 

            # Get the name to save the information.
            if Holes:
                folder = 'Holes/' + size + '/'
            else:
                folder = 'Clouds/' + size + '/'
            
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Generate the Graph of the Cloud
            CloudGen.GraphCloud(p, save = True, folder = folder, nom = reg, show = False)

            # Save the information to CSV files.
            pd.DataFrame(p).to_csv(f'{folder}{reg}_p.csv', index = False, header = False)
            pd.DataFrame(tt).to_csv(f'{folder}{reg}_tt.csv', index = False, header = False)

config = [(True), (False)]

for Holes in config:
    run(Holes)