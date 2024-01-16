import train_nets as tn

# 0000 0000 0000 0000 1101 0000 0011 0000
fixed_bits_map = {
  4: 1,
  5: 1,
  6: 0,
  7: 0,
  12: 1,
  13: 0,
  14: 1,
  15: 1
}
tn.train_intg_speck_distinguisher(200, fixed_bits_map=fixed_bits_map, num_rounds=7,depth=1,reg_param=10**-6)
