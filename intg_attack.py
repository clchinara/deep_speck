import random
import numpy as np

def generate_template(num_unfixed_bits):
    template = np.zeros(32, dtype=np.uint8)
    fixed_bit_positions = np.random.choice(np.arange(32), num_unfixed_bits, replace=False)
    template[fixed_bit_positions] = 1
    return template

def generate_plaintexts_from_template(template, num_samples):
    num_unfixed_bits = np.sum(template == 0)
    unfixed_combinations = np.random.randint(2, size=(2 ** num_unfixed_bits, num_unfixed_bits), dtype=np.uint8)
    plaintexts = np.zeros((2 ** num_unfixed_bits, 32), dtype=np.uint8)
    plaintexts[:, template == 1] = 1 # for now, only assign 1s at the fixed positions
    plaintexts[:, template == 0] = unfixed_combinations
    return np.tile(plaintexts, (num_samples, 1))

def generate_integral_plaintexts(num_samples, num_unfixed_bits):
    # Generate templates
    templates = [generate_template(num_unfixed_bits) for _ in range(num_samples)]
    # Generate plaintexts for each template, i.e. for each sample
    plaintexts_per_template = [generate_plaintexts_from_template(template, 1) for template in templates]
    plaintexts_per_template_np = np.array(plaintexts_per_template) # (num_samples, 2^num_unfixed_bits, 32)
    # Separate left & right part
    left_part = np.packbits(plaintexts_per_template_np[:, :, :16].reshape(-1, num_samples, 2, 8)[:, :, ::-1]).view(np.uint16).reshape(plaintexts_per_template_np.shape[:2]) # (num_samples, 2^num_unfixed_bits)
    right_part = np.packbits(plaintexts_per_template_np[:, :, 16:].reshape(-1, num_samples, 2, 8)[:, :, ::-1]).view(np.uint16).reshape(plaintexts_per_template_np.shape[:2]) # (num_samples, 2^num_unfixed_bits)
    return np.transpose(left_part), np.transpose(right_part) # (2^num_unfixed_bits, num_samples) and (2^num_unfixed_bits, num_samples)

def generate_integral_plaintexts_depr(Y, fixedBitPositionToValue, numPlaintexts):
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