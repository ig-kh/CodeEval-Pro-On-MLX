

def how_many_times(string: str, substring: str) -> int:
    """ Find how many times a given substring can be found in the original string. Count overlaping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """
    import re

    count = 0
    start = 0

    while True:
        index = string.find(substring, start)
        if index == -1:
            break
        count += 1
        start = index + 1  # Move only one character forward to allow overlapping matches

    return count


def total_substring_counts(strings: list, substrings: list) -> dict:
    """ Given a list of strings and a list of substrings, find the total number of times each substring appears in the list of strings. Return a dictionary where the keys are the substrings and the values are the total counts across all strings.
    """
    counts = {}
    for s in strings:
        for sub in substrings:
            count = how_many_times(s, sub)
            counts[sub] = count + counts.get(sub, 0)

    return counts


assert total_substring_counts(['aaa', 'aaaa'], ['a', 'aa']) == {'a': 7, 'aa': 5}
assert total_substring_counts(['', 'test'], ['t', 'st']) == {'t': 2, 'st': 1}
assert total_substring_counts(['hello', 'world'], ['l', 'o']) == {'l': 3, 'o': 2}