import train_nets as tn

# 0000 0000 0000 0000 1101 0000 0011 0000
plaintext_template = 'xxxxxxxxxxxxxxxx1100101010110010'
assert len(plaintext_template) == 32
fixed_bits_map = {}
for i, bit in enumerate(plaintext_template):
  if bit != 'x':
    fixed_bits_map[i] = int(bit)
print('fixed_bits_map:', fixed_bits_map)
tn.train_intg_speck_distinguisher(200, fixed_bits_map=fixed_bits_map, num_rounds=9,depth=1,reg_param=10**-6)

