import speck as sp
import numpy as np
import constants as cs

from keras.models import load_model

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

with open('inv2_logs.txt', 'w') as fn:
    for diff in cs.DIFFS:
        net7 = load_model(f'./inv2_freshly_trained_nets/{diff}/best7depth10.h5')
        nets = [net7]
        fn.write(f'========== DIFF: {diff} ==========\n')
        for i, net in enumerate(nets):
            num_rounds = i + 7
            X, Y = sp.make_train_data(10**6, num_rounds, diff)
            fn.write(f'{num_rounds} rounds:\n{evaluate(net, X, Y)}\n')
fn.close()
