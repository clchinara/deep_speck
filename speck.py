import numpy as np
from os import urandom
from collections import Counter

from constants import INV_NAME, DIFF_B, DIFF_C, NUM_PLAINTEXTS
import intg_attack as intg

def WORD_SIZE():
    return(16); # 16-bit

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    return(False);

#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros(((NUM_PLAINTEXTS * 2) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range((NUM_PLAINTEXTS * 2) * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator
def make_train_data(n, nr, diffA=(0x0040,0), diffB=DIFF_B, diffC=DIFF_C):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); # 16-bit
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  print('plain0l', plain0l)
  print('plain0l.shape', plain0l.shape)
  plain1l = plain0l ^ diffA[0]; plain1r = plain0r ^ diffA[1];
  plain2l = plain1l ^ diffB[0]; plain2r = plain1r ^ diffB[1];
  plain3l = plain2l ^ diffA[0]; plain3r = plain2r ^ diffA[1];
  plain4l = plain0l ^ diffC[0]; plain4r = plain0r ^ diffC[1];
  plain5l = plain4l ^ diffA[0]; plain5r = plain4r ^ diffA[1];
  plain6l = plain5l ^ diffB[0]; plain6r = plain5r ^ diffB[1];
  plain7l = plain6l ^ diffA[0]; plain7r = plain6r ^ diffA[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain2l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain2r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain3l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain3r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain4l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain4r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain5l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain5r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain6l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain6r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain7l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain7r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks);
  ctdata4l, ctdata4r = encrypt((plain4l, plain4r), ks);
  ctdata5l, ctdata5r = encrypt((plain5l, plain5r), ks);
  ctdata6l, ctdata6r = encrypt((plain6l, plain6r), ks);
  ctdata7l, ctdata7r = encrypt((plain7l, plain7r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r, ctdata2l, ctdata2r, ctdata3l, ctdata3r, ctdata4l, ctdata4r, ctdata5l, ctdata5r, ctdata6l, ctdata6r, ctdata7l, ctdata7r])
  # X.shape = (1000, 256) where n = 1000
  # Y.shape = (1000, )
  return(X,Y);

def make_intg_train_data(n, nr, fixedBitPositionToValue, isMultiset, numPlaintexts=NUM_PLAINTEXTS):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
  ks = expand_key(keys, nr)
  n_plaintexts = []
  for _ in range(n):
    plaintexts = intg.generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, numPlaintexts)
    n_plaintexts.append(plaintexts)
  n_plaintexts_np = np.vstack(n_plaintexts)
  ctdata = []
  print('indices', np.where(Y == 0))
  print('len(indices)', len(np.where(Y == 0)[0]))
  for i in range(numPlaintexts):
    plaini = n_plaintexts_np[:, i] # (n, )
    plainil, plainir = intg.split_32bit_to_16bit(plaini)
    num_rand_samples = np.sum(Y==0)
    plainil[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    plainir[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16)
    if i == 0:
    #   print('plainil:', plainil)
    #   print('plainil.shape:', plainil.shape)
      print('plaini[0]', plaini[0])
      print('plainil[0]', plainil[0])
      print('plainir[0]', plainir[0])
      print('plaini:', intg.number_to_np_binary_string(plaini[0]))
      print('plainil:', intg.number_to_np_binary_string(plainil[0])[16:])
      print('plainir:', intg.number_to_np_binary_string(plainir[0])[16:])
    ctdatail, ctdatair = encrypt((plainil, plainir), ks)
    ctdata += [ctdatail, ctdatair]
  X = convert_to_binary(ctdata)
  return (X, Y)

#real differences data generator
def real_differences_data(n, nr, diff=(0x0040,0)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #apply input difference
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  #expand keys and encrypt
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

def get_diffs_after_nr_rounds(n, nr=3, diffA=(0x0040,0), diffB=DIFF_B, diffC=DIFF_C):
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16); # 16-bit
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diffA[0]; plain1r = plain0r ^ diffA[1];
  plain2l = plain1l ^ diffB[0]; plain2r = plain1r ^ diffB[1];
  plain3l = plain2l ^ diffA[0]; plain3r = plain2r ^ diffA[1];
  plain4l = plain0l ^ diffC[0]; plain4r = plain0r ^ diffC[1];
  plain5l = plain4l ^ diffA[0]; plain5r = plain4r ^ diffA[1];
  plain6l = plain5l ^ diffB[0]; plain6r = plain5r ^ diffB[1];
  plain7l = plain6l ^ diffA[0]; plain7r = plain6r ^ diffA[1];
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  ctdata2l, ctdata2r = encrypt((plain2l, plain2r), ks);
  ctdata3l, ctdata3r = encrypt((plain3l, plain3r), ks);
  ctdata4l, ctdata4r = encrypt((plain4l, plain4r), ks);
  ctdata5l, ctdata5r = encrypt((plain5l, plain5r), ks);
  ctdata6l, ctdata6r = encrypt((plain6l, plain6r), ks);
  ctdata7l, ctdata7r = encrypt((plain7l, plain7r), ks);

  diffs_c0_c1 = []
  diffs_c1_c2 = []
  diffs_c1_c5 = []
  diffs_c2_c3 = []
  diffs_c0_c3 = []
  diffs_c2_c6 = []
  diffs = []

  for i in range(n):
    diff_c0_c1 = (ctdata0l[i] ^ ctdata1l[i], ctdata0r[i] ^ ctdata1r[i])
    diff_c1_c2 = (ctdata1l[i] ^ ctdata2l[i], ctdata1r[i] ^ ctdata2r[i])
    diff_c1_c5 = (ctdata1l[i] ^ ctdata5l[i], ctdata1r[i] ^ ctdata5r[i])
    diff_c2_c3 = (ctdata2l[i] ^ ctdata3l[i], ctdata2r[i] ^ ctdata3r[i])
    diff_c0_c3 = (ctdata0l[i] ^ ctdata3l[i], ctdata0r[i] ^ ctdata3r[i])
    diff_c2_c6 = (ctdata2l[i] ^ ctdata6l[i], ctdata2r[i] ^ ctdata6r[i])
    diffs_c0_c1.append(diff_c0_c1)
    diffs_c1_c2.append(diff_c1_c2)
    diffs_c1_c5.append(diff_c1_c5)
    diffs_c2_c3.append(diff_c2_c3)
    diffs_c0_c3.append(diff_c0_c3)
    diffs_c2_c6.append(diff_c2_c6)
    diffs.append((diff_c0_c1, diff_c1_c2, diff_c1_c5, diff_c2_c3, diff_c0_c3, diff_c2_c6)) # 2

  diffs_cnt = Counter(diffs)
  sorted_diffs_cnt = dict(sorted(diffs_cnt.items(), key=lambda item: item[1], reverse=True))
  top_10_diffs_cnt = dict(list(sorted_diffs_cnt.items())[:10])
  print(top_10_diffs_cnt)

  with open(f'{INV_NAME}_diff_after_{nr}_rounds1.txt', 'w') as fn:
    res = most_occuring(diffs_c0_c1), most_occuring(diffs_c1_c2), most_occuring(diffs_c1_c5), \
        most_occuring(diffs_c2_c3), most_occuring(diffs_c0_c3), most_occuring(diffs_c2_c6), \
        most_occuring(diffs)
    fn.write(f'diff_c0_c1: {res[0]}\n')
    fn.write(f'diff_c1_c2: {res[1]}\n')
    fn.write(f'diff_c1_c5: {res[2]}\n')
    fn.write(f'diff_c2_c3: {res[3]}\n')
    fn.write(f'diff_c0_c3: {res[4]}\n')
    fn.write(f'diff_c2_c6: {res[5]}\n')
    fn.write(f'diff: {res[6]}\n')
    fn.write(f'top_10_diffs_cnt: {top_10_diffs_cnt}\n')
  fn.close()

#   print('diffs_c0_c1:', diffs_c0_c1)
#   print('diffs_c1_c2:', diffs_c1_c2)
#   print('diffs_c2_c3:', diffs_c2_c3)
#   print('diffs_c3_c0:', diffs_c3_c0)
#   print('ctdata0l:', ctdata0l)
#   print('ctdata0l[0]:', ctdata0l[0])
#   print('ctdata0r:', ctdata0r)
#   print('ctdata0l.shape:', ctdata0l.shape)
#   print('ctdata0r.shape:', ctdata0r.shape)

  return res

def most_occuring(lst):
    element = max(set(lst), key=lst.count)
    count = lst.count(element)
    return element, count