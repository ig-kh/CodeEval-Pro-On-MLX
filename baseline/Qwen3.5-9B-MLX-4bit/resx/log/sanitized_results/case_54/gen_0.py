

def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
    from collections import Counter
    return Counter(s0) == Counter(s1)


def can_form_by_rearranging(words: list) -> list:
    """
    Given a list of words, determine if each word can be formed by rearranging the characters of any other word in the list. Return a list of boolean values indicating whether each word can be formed by rearranging the

assert can_form_by_rearranging(['eabcdzzzz', 'dddzzzzzzzddeddabc', 'abcd', 'dddddddabc', 'eabcd', 'dddddddabce', 'abcd', 'dddddddabce']) == [True, True, True, True, True, True, True, True]
assert can_form_by_rearranging(['abcd', 'dcba', 'abcd', 'abcd']) == [True, True, True, True]
assert can_form_by_rearranging(['abc', 'def', 'ghi', 'jkl']) == [False, False, False, False]
assert can_form_by_rearranging(['abc', 'cba', 'bca', 'acb']) == [True, True, True, True]