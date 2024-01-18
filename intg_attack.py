import random
import numpy as np

def generate_integral_plaintexts(Y, fixedBitPositionToValue, numPlaintexts):
    num_data = Y.shape[0]
    bit_array = np.random.choice([0, 1], size=(numPlaintexts, num_data, 32))
    # print(bit_array[0])
    # print(bit_array[:, :16][0])
    # print(bit_array[:, 16:][0])
    # print(bit_array.shape)

    assignments_array = np.zeros(32, dtype=np.uint8)
    assignments_mask = np.zeros(32, dtype=np.bool_)
    for position in fixedBitPositionToValue:
        assignments_array[position] = fixedBitPositionToValue[position]
        assignments_mask[position] = True
    # print('assignments_array', assignments_array)
    # print('assignments_mask', assignments_mask)
        
    # Broadcast the assignments to bit_array according to mask
    Y_mask = np.expand_dims(np.expand_dims(Y != 0, axis=0), axis=2)
    expanded_mask = np.tile(Y_mask, (1, 1, 32))
    # print('expanded_mask.shape', expanded_mask.shape)
    expanded_mask[:, :, :] = np.where(Y_mask, assignments_mask, expanded_mask[:, :, :])
    # print('expanded_mask', expanded_mask)
    bit_array = np.where(expanded_mask, np.broadcast_to(assignments_array, bit_array.shape), bit_array)
    # print(bit_array[0])

    left_part = np.packbits(bit_array[:, :, :16].reshape(-1, num_data, 2, 8)[:, :, ::-1]).view(np.uint16).reshape(bit_array.shape[:2])
    right_part = np.packbits(bit_array[:, :, 16:].reshape(-1, num_data, 2, 8)[:, :, ::-1]).view(np.uint16).reshape(bit_array.shape[:2])

    return left_part, right_part # (numPlaintexts, numData) and (numPlaintexts, numData)

def generate_integral_plaintexts1(fixedBitPositionToValue, isMultiset, numPlaintexts):
    # random_bits_len = 32 - len(fixedBitPositionToValue)
    plaintexts = [0] * numPlaintexts

    # for _ in range(min(2 ** random_bits_len, numPlaintexts)):
    for i in range(numPlaintexts):
        random_bits = np.random.choice([0, 1], size=32)
        plaintext_number = 0
        for j in range(32):
            if j in fixedBitPositionToValue:
                plaintext_number |= fixedBitPositionToValue[j] << j
            else:
                plaintext_number |= random_bits[j] << j
        plaintexts[i] = plaintext_number

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