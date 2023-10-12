
def parse_hex_string(hex_string):
    return int(hex_string, 16)

def extract_left_right(number):
    # Convert the hexadecimal number to a binary string
    binary_string = bin(number)[2:]
    # Ensure the binary string is 32 bits long by padding with leading zeros if needed
    binary_string = binary_string.zfill(32)
    # Split the binary string into two 16-bit parts (left and right)
    left_part = binary_string[:16]
    right_part = binary_string[16:]

    left_val = int(left_part, 2)
    right_val = int(right_part, 2)
    return left_val, right_val

def combine_left_right(left, right):
    return (left << 16) | right