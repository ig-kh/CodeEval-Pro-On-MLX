
def is_happy(s):
    """You are given a string s.
    Your task is to check if the string is happy or not.
    A string is happy if its length is at least 3 and every 3 consecutive letters are distinct
    For example:
    is_happy(a) => False
    is_happy(aa) => False
    is_happy(abcd) => True
    is_happy(aabb) => False
    is_happy(adb) => True
    is_happy(xyy) => False
    """
    if len(s) < 3:
        return False
    
    for i in range(len(s) - 2):
        if s[i] == s[i+1] or s[i] == s[i+2] or s[i+1] == s[i+2]:
            return False
    
    return True

def count_happy_strings(strings):
    """You are given a list of strings. Your task is to determine the number of happy strings in the list. A string is happy if its length is at least 3 and every 3 consecutive letters are distinct. You need to implement a function `count_happy_strings(strings)` that returns the count of happy strings in the input list."""
    return sum(1 for s in strings if is_happy(s))


assert count_happy_strings(['a', 'aa', 'abcd', 'aabb', 'adb', 'xyy']) == 2
assert count_happy_strings(['abc', 'def', 'ghi', 'jkl']) == 4
assert count_happy_strings(['aaa', 'bbb', 'ccc', 'ddd']) == 0
assert count_happy_strings(['xyz', 'uvw', 'rst', 'opq']) == 4
assert count_happy_strings(['a', 'ab', 'abc', 'abcd', 'abcde']) == 3