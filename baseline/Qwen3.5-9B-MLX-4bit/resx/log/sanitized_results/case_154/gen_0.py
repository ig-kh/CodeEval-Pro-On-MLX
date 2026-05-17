
def cycpattern_check(a , b):
    """You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word
    cycpattern_check("abcd","abd") => False
    cycpattern_check("hello","ell") => True
    cycpattern_check("whassup","psus") => False
    cycpattern_check("abab","baa") => True
    cycpattern_check("efef","eeff") => False
    cycpattern_check("himenss","simen") => True

    """
    import re

    if len(a) < len(b):
        return False

    for i in range(len(b)):
        if b[i:] + b[:i] in a:
            return True
    return False

def valid_rotations(words):
    """Given a list of words, determine if each word in the list can be transformed into a valid rotation of any other word in the list. A valid rotation means that if you rotate the word any number of times, it will become a substring of another word in the list. Return a list of boolean values indicating whether each word can be transformed into a valid rotation of any other word in the list.
    """
    result = []
    for word in words:
        if any(cycpattern_check(word, other) for other in words):
            result.append(True)
        else:
            result.append(False)
    return result


assert valid_rotations(['abcd', 'abd']) == [False, False]
assert valid_rotations(['hello', 'ell']) == [False, True]
assert valid_rotations(['whassup', 'psus']) == [False, False]
assert valid_rotations(['abab', 'baa']) == [False, True]
assert valid_rotations(['efef', 'eeff']) == [False, False]
assert valid_rotations(['himenss', 'simen']) == [False, True]