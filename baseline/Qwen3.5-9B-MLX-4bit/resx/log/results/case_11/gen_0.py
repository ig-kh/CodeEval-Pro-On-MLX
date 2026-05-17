from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """
    # Ensure both strings are of the same length
    if len(a) != len(b):
        raise ValueError("Strings must be of equal length for XOR operation.")
    
    result = []
    for char_a, char_b in zip(a, b):
        if char_a == '0' and char_b == '1':
            result.append('1')
        elif char_a == '1' and char_b == '0':
            result.append('1')
        else:
            result.append('0')
    
    return ''.join(result)


def cumulative_xor(binary_strings: List[str]) -> str:
    """ Given a list of binary strings, perform a cumulative XOR operation on them.
    The cumulative XOR operation means that for each string in the list, XOR it with the result of all previous XOR operations.
    Return the final result as a string.
    """
    if not binary_strings:
        return ""
    
    result = binary_strings[0]
    for s in binary_strings[1:]:
        result = string_xor(result, s)
    
    return result


assert cumulative_xor(['010', '110', '101']) == '001'
assert cumulative_xor(['111', '111', '111']) == '111'
assert cumulative_xor(['000', '000', '000']) == '000'
assert cumulative_xor(['101', '010', '111']) == '000'
assert cumulative_xor(['110', '011', '100']) == '001'