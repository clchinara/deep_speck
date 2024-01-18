import random
import numpy as np

# def generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, numPlaintexts):
#     fixed_positions = list(fixedBitPositionToValue.keys())
#     fixed_values = list(fixedBitPositionToValue.values())

#     unfixed_bits = 32 - len(fixed_positions)

#     # Generate plaintexts by combining fixed and unfixed bits using bit manipulation
#     plaintexts = []
#     for i in range(min(2 ** unfixed_bits, numPlaintexts)):
#         plaintext_number = 0
#         unfixed_indices = [j for j in range(32) if j not in fixed_positions]

#         random.shuffle(unfixed_indices)

#         for j in range(32):
#             if j in fixed_positions:
#                 plaintext_number |= fixed_values[fixed_positions.index(j)] << j
#             else:
#                 # plaintext_number |= ((i >> unfixed_indices.pop(0)) & 1) << j
#                 plaintext_number |= (random.choice([0, 1])) << j

#         plaintexts.append(plaintext_number)

#     if isMultiset:
#         return plaintexts
#     else:
#         plaintext_set = set(plaintexts)
#         plaintext_set_length = len(plaintext_set)
#         if plaintext_set_length == numPlaintexts:
#             return plaintexts
#         while plaintext_set_length != numPlaintexts:
#           remaining_num_plaintexts = numPlaintexts - len(plaintext_set)
#           remaining_plaintexts = generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, remaining_num_plaintexts)
#           plaintext_set = plaintext_set.union(set(remaining_plaintexts))
#         return list(plaintext_set)


def generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, numPlaintexts):
    # random_bits_len = 32 - len(fixedBitPositionToValue)
    random_bits = np.random.choice([0, 1], size=32)

    plaintexts = []

    # for _ in range(min(2 ** random_bits_len, numPlaintexts)):
    for _ in range(numPlaintexts):
        plaintext_number = 0
        for j in range(32):
            if j in fixedBitPositionToValue:
                plaintext_number |= fixedBitPositionToValue[j] << j
            else:
                plaintext_number |= random_bits[j] << j

        plaintexts.append(plaintext_number)

    if isMultiset:
        return plaintexts

    remaining_plaintexts_needed = numPlaintexts - len(plaintexts)

    while remaining_plaintexts_needed > 0:
        batch_size = min(remaining_plaintexts_needed, 10000)
        remaining_plaintexts = []

        for _ in range(batch_size):
            plaintext_number = 0

            for j in range(32):
                plaintext_number |= (np.random.choice([0, 1])) << j

            remaining_plaintexts.append(plaintext_number)

        plaintexts.extend(remaining_plaintexts)
        remaining_plaintexts_needed -= batch_size

    return plaintexts[:numPlaintexts]

def number_to_np_binary_string(number):
    binary_string = format(number, '032b')
    binary_array = np.array(list(map(int, binary_string)))
    return binary_array

def split_32bit_to_16bit(original_array):
    original_array = np.asarray(original_array, dtype=np.uint32)
    right_16_bits = original_array & 0xFFFF
    left_16_bits = (original_array >> 16) & 0xFFFF
    return left_16_bits, right_16_bits