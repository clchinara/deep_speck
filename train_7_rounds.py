import train_nets as tn
from constants import DIFFS

for diff in DIFFS:
    tn.train_speck_distinguisher(200,num_rounds=7,depth=1, diff=diff, subdir=f'{diff}/');