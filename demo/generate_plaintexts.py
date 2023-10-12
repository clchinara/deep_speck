import utils

p0 = 0x5b024900
diffA = (0x0040, 0)
diffB = (0x0014, 0x0800)
plain0l, plain0r = utils.extract_left_right(p0)
plain1l = plain0l ^ diffA[0]; plain1r = plain0r ^ diffA[1];
plain2l = plain1l ^ diffB[0]; plain2r = plain1r ^ diffB[1];
plain3l = plain2l ^ diffA[0]; plain3r = plain2r ^ diffA[1];
plain0 = utils.combine_left_right(plain0l, plain0r)
plain1 = utils.combine_left_right(plain1l, plain1r)
plain2 = utils.combine_left_right(plain2l, plain2r)
plain3 = utils.combine_left_right(plain3l, plain3r)
print(hex(plain0)[2:], hex(plain1)[2:], hex(plain2)[2:], hex(plain3)[2:])