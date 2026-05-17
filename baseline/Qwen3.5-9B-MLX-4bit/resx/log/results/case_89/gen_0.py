
def encrypt(s):
    """Create a function encrypt that takes a string as an argument and
    returns a string encrypted with the alphabet being rotated. 
    The alphabet should be rotated in a manner such that the letters 
    shift down by two multiplied to two places.
    For example:
    encrypt('hi') returns 'lm'
    encrypt('asdfghjkl') returns 'ewhjklnop'
    encrypt('gf') returns 'kj'
    encrypt('et') returns 'ix'
    """
    result = []
    for char in s:
        if 'a' <= char <= 'z':
            result.append(chr((ord(char) - ord('a') + 4) % 26 + ord('a')))
        elif 'A' <= char <= 'Z':
            result.append(chr((ord(char) - ord('A') + 4) % 26 + ord('A')))
        else:
            result.append(char)
    return ''.join(result)

def encrypt_message(strings, rotate_by=2):
    """Create a function `encrypt_message` that takes a list of strings as an argument and returns a list of strings where each string is encrypted using the `encrypt

assert encrypt_message(['hi', 'asdfghjkl', 'gf', 'et']) == ['lm', 'ewhjklnop', 'kj', 'ix']
assert encrypt_message(['hi', 'asdfghjkl', 'gf', 'et'], 3) == ['no', 'gyjlmnpqr', 'ml', 'kz']
assert encrypt_message(['abc', 'xyz'], 1) == ['cde', 'zab']
assert encrypt_message(['hello', 'world'], 0) == ['hello', 'world']