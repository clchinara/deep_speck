import speck as sp
import my_speck as my_sp
import numpy as np

from keras.models import model_from_json

#load distinguishers
json_file = open('single_block_resnet.json','r');
json_model = json_file.read();

net5 = model_from_json(json_model);
net6 = model_from_json(json_model);
net7 = model_from_json(json_model);
net8 = model_from_json(json_model);

net5.load_weights('net5_small.h5');
net6.load_weights('net6_small.h5');
net7.load_weights('net7_small.h5');
net8.load_weights('net8_small.h5');

def evaluate(net,X,Y):
    Z = net.predict(X,batch_size=10000).flatten();
    Zbin = (Z > 0.5);
    diff = Y - Z; mse = np.mean(diff*diff);
    n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    acc = np.sum(Zbin == Y) / n;
    tpr = np.sum(Zbin[Y==1]) / n1;
    tnr = np.sum(Zbin[Y==0] == 0) / n0;
    mreal = np.median(Z[Y==1]);
    high_random = np.sum(Z[Y==0] > mreal) / n0;
    log_str = f'Accuracy: {acc}, TPR: {tpr}, TNR: {tnr}, MSE: {mse}\nPercentage of random pairs with score higher than median of real pairs: {100*high_random}'
    print("Accuracy: ", acc, "TPR: ", tpr, "TNR: ", tnr, "MSE:", mse);
    print("Percentage of random pairs with score higher than median of real pairs:", 100*high_random);
    return log_str

# DIFFS = [(0x0040,0), (0x211a04,0), (0x0A604205,0), (0x8054A900,0), (0x20400040,0), (0x02110A04,0), (0x00140800,0), (0x14AC5209,0)]

# with open('res_logs.txt', 'w') as fn:

#     for diff in DIFFS:
#         X5,Y5 = sp.make_train_data(10**6,5,diff);
#         X6,Y6 = sp.make_train_data(10**6,6,diff);
#         X7,Y7 = sp.make_train_data(10**6,7,diff);
#         X8,Y8 = sp.make_train_data(10**6,8,diff);

#         X5r, Y5r = sp.real_differences_data(10**6,5,diff);
#         X6r, Y6r = sp.real_differences_data(10**6,6,diff);
#         X7r, Y7r = sp.real_differences_data(10**6,7,diff);
#         X8r, Y8r = sp.real_differences_data(10**6,8,diff);

#         fn.write(f'========== Diff: {diff} ==========\n\n')

#         print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
#         fn.write('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting\n')
#         print('5 rounds:');
#         fn.write('5 rounds:\n')
#         fn.write(f'{evaluate(net5, X5, Y5)}\n');
#         print('6 rounds:');
#         fn.write('6 rounds:\n')
#         fn.write(f'{evaluate(net6, X6, Y6)}\n');
#         print('7 rounds:');
#         fn.write('7 rounds:\n')
#         fn.write(f'{evaluate(net7, X7, Y7)}\n');
#         print('8 rounds:');
#         fn.write('8 rounds:\n')
#         fn.write(f'{evaluate(net8, X8, Y8)}\n\n');

#         print('\nTesting real differences setting now.');
#         fn.write('Testing real differences setting now\n')
#         print('5 rounds:');
#         fn.write('5 rounds:\n')
#         fn.write(f'{evaluate(net5, X5r, Y5r)}\n');
#         print('6 rounds:');
#         fn.write('6 rounds:\n')
#         fn.write(f'{evaluate(net6, X6r, Y6r)}\n');
#         print('7 rounds:');
#         fn.write('7 rounds:\n')
#         fn.write(f'{evaluate(net7, X7r, Y7r)}\n');
#         print('8 rounds:');
#         fn.write('8 rounds:\n')
#         fn.write(f'{evaluate(net8, X8r, Y8r)}\n\n');

# fn.close()

"""
Looking into diffB = (0x20400040,0)
"""

nets = [net5, net6, net7, net8]
ROUNDS = [5, 6, 7, 8]
X_LABELS = ['X_01', 'X_02', 'X_03', 'X_12', 'X_13', 'X_23'] # X_01 contains ciphertext pairs 0 and 1, and so on

with open('res_logs_1.txt', 'w') as fn:

    # Normal setting only
    print('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting');
    fn.write('Testing neural distinguishers against 5 to 8 blocks in the ordinary real vs random setting\n\n')

    for i, rnd in enumerate(ROUNDS):
        X, Y = my_sp.make_train_data(10**6, rnd, diffA=(0x0040,0), diffB=(0x20400040,0))
        print(f'{rnd} rounds:')
        fn.write(f'{rnd} rounds:\n')

        # Unpack X (which is a tuple) & evaluate each of (X_01, X_02, X_03, X_12, X_13, X_23)
        for j, x in enumerate(X):
            fn.write(f'===== Using X = {X_LABELS[j]} =====\n')
            fn.write(f'{evaluate(nets[i], x, Y)}\n')
        
        # Record combined results i.e. where all pairings (which are contained in X) are evaluated at once
        fn.write('===== Combined results (6n records) =====\n')
        x_all = np.concatenate(X, axis=0)
        y_all = np.tile(Y, len(X))
        fn.write(f'{evaluate(nets[i], x_all, y_all)}\n')

        fn.write('\n')

fn.close()