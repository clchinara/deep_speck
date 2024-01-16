import random
import numpy as np

def generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, numPlaintexts):
    fixed_positions = list(fixedBitPositionToValue.keys())
    fixed_values = list(fixedBitPositionToValue.values())

    unfixed_bits = 32 - len(fixed_positions)

    # Generate plaintexts by combining fixed and unfixed bits using bit manipulation
    plaintexts = []
    for i in range(min(2 ** unfixed_bits, numPlaintexts)):
        plaintext_number = 0
        unfixed_indices = [j for j in range(32) if j not in fixed_positions]

        random.shuffle(unfixed_indices)

        for j in range(32):
            if j in fixed_positions:
                plaintext_number |= fixed_values[fixed_positions.index(j)] << j
            else:
                # plaintext_number |= ((i >> unfixed_indices.pop(0)) & 1) << j
                plaintext_number |= (random.choice([0, 1])) << j

        plaintexts.append(plaintext_number)

    if isMultiset:
        return plaintexts
    else:
        plaintext_set = set(plaintexts)
        plaintext_set_length = len(plaintext_set)
        if plaintext_set_length == numPlaintexts:
            return plaintexts
        while plaintext_set_length != numPlaintexts:
          remaining_num_plaintexts = numPlaintexts - len(plaintext_set)
          remaining_plaintexts = generate_integral_plaintexts(fixedBitPositionToValue, isMultiset, remaining_num_plaintexts)
          plaintext_set = plaintext_set.union(set(remaining_plaintexts))
        return list(plaintext_set)

def number_to_np_binary_string(number):
    binary_string = format(number, '032b')
    binary_array = np.array(list(map(int, binary_string)))
    return binary_array
