import os
import sys
sys.path.append(os.getcwd())
# import pandas as pd

# jz_description_file = 'data/sample_description.csv'
# path = '/scratch/ucjf-atlas/jankovys/JIDENN/data/all/'
new_path = '/scratch/ucjf-atlas/jankovys/JIDENN/data/all_MCs/'

# jz_description = pd.read_csv(jz_description_file, index_col=0)
# jz_description.index = jz_description.index.astype(str)

# for dssid in os.listdir(path):
#     jz = jz_description.loc[dssid]['JZ']
#     name = jz_description.loc[dssid]['Description']
#     name = name[:name.rfind('_')]
#     os.makedirs(os.path.join(new_path, name, f'JZ{jz}'), exist_ok=True)
#     os.rename(os.path.join(path, dssid), os.path.join(
#         new_path, name, f'JZ{jz}', dssid))

for MC in os.listdir(new_path):
    for JZ in os.listdir(os.path.join(new_path, MC)):
        inside = os.listdir(os.path.join(new_path, MC, JZ))
        if len(inside) == 1:
            os.system(f'mv {os.path.join(new_path, MC, JZ, inside[0])}/* {os.path.join(new_path, MC, JZ)}')
            os.system(f'rmdir {os.path.join(new_path, MC, JZ, inside[0])}')