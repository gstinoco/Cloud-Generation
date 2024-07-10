import os
import glob
import CloudGen
import pandas as pd

def run_randomize(Holes):
    # Variable Initialization.
    sizes = ['1', '2', '3']

    for me in sizes:
        # Find all the regions.
        if Holes:
            folder = 'Holes/' + me + '/'
        else:
            folder = 'Clouds/' + me + '/'
        regions = glob.glob(folder + '*_p.csv')
        regions = sorted([os.path.splitext(os.path.basename(region))[0].replace('_p', '') for region in regions])

        for reg in regions:
            # Load data from the files.
            p = pd.read_csv(folder + reg + '_p.csv').values
            tt = pd.read_csv(folder + reg + '_tt.csv').values

            mesh = str(me)
            print(f'Working on {reg}_{mesh}.')

            if Holes:
                folder2 = 'Holes_rand/' + me + '/'
            else:
                folder2 = 'Clouds_rand/' + me + '/'
            
            if not os.path.exists(folder2):
                os.makedirs(folder2)

            # Generate the Cloud
            p = CloudGen.Randomize(p, me)

            CloudGen.GraphCloud(p, save = True, folder = folder2, nom = reg, show = False)

            # Save the information of the Cloud.
            pd.DataFrame(p).to_csv(f'{folder2}{reg}_p.csv', index = False, header = False)
            pd.DataFrame(tt).to_csv(f'{folder2}{reg}_tt.csv', index = False, header = False)

config = [(True), (False)]

for Holes in config:
    run_randomize(Holes)