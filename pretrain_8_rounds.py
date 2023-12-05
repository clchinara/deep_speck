import speck as sp
import train_nets as tn

import math
from keras.models import load_model

from constants import INV_NAME

top_diff_cnt = {((32772, 33806), (32960, 33216), (32772, 33806), (32960, 33216)): 26, ((32772, 33806), (32832, 33088), (32772, 33806), (32832, 33088)): 28, ((33280, 34314), (32832, 33088), (33280, 34314), (32832, 33088)): 35, ((32768, 33802), (32960, 33216), (32768, 33802), (32960, 33216)): 43, ((32768, 33802), (32832, 33088), (32768, 33802), (32832, 33088)): 88}
# sorted_top_diff_cnt = dict(sorted(top_diff_cnt.items(), key=lambda item: item[1], reverse=False))

total_cnt = sum(top_diff_cnt.values())
total_num_training_data = 10**7

# diff_to_num_train_data = dict()

# for i, (diff, cnt) in enumerate(top_diff_cnt.items()):
#     diffA = diff[0]
#     diffB = diff[1]
#     if i == len(top_diff_cnt) - 1:
#         curr_total = sum(diff_to_num_train_data.values())
#         diff_to_num_train_data[(diffA, diffB)] = total_num_training_data - curr_total
#     else:
#         diff_to_num_train_data[(diffA, diffB)] = math.floor(cnt / total_cnt * total_num_training_data)


# print('diff_to_num_train_data:', diff_to_num_train_data)
# print('sum(diff_to_num_train_data.values()):', sum(diff_to_num_train_data.values()))

model_dir = f'{INV_NAME}_freshly_trained_nets'

net7 = load_model(f'{model_dir}/best7depth1.h5')
# tn.pretrain_speck_distinguisher_8_rounds_new(net7=net7, diff_to_num_train_data=diff_to_num_train_data, num_epochs=20)

# Get top 7
diffPairs = list(top_diff_cnt.keys())[:7]
diffAs = []
diffBs = []
for diffPair in diffPairs:
    diffAs.append(diffPair[0])
    diffBs.append(diffPair[1])

tn.pretrain_speck_distinguisher_8_rounds(net7=net7, diffAs=diffAs, diffBs=diffBs, num_epochs=20)