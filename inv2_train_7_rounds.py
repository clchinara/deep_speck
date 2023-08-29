import inv2_train_nets as inv2_tn
import constants as cs

for diff in cs.DIFFS:
    inv2_tn.train_speck_distinguisher(200,num_rounds=7,depth=10, diff=diff, subdir=f'{diff}/');