

def count_distinct_characters(string: str) -> int:
    """ Given a string, find out how many distinct characters (regardless of case) does it consist of
    >>> count_distinct_characters('xyzXYZ')
    3
    >>> count_distinct_characters('Jerry')
    4
    """
    char_set = set()
    for char in string:
        char_set.add(char.lower())
    
    return len(char_set)


def count_distinct_characters_across_strings(strings: list) -> int:
    """ Given a list of strings, find out how many distinct characters (regardless of case) are present across all strings.
    """
    if not strings:
        return 0
    
    char_set = set()
    for s in strings:
        for char in s:
            char_set.add(char.lower())
    
    return len(char_set)


assert count_distinct_characters_across_strings(['xyzXYZ', 'Jerry']) == 6
assert count_distinct_characters_across_strings(['Hello', 'world']) == 7
assert count_distinct_characters_across_strings(['Python', 'java']) == 9
assert count_distinct_characters_across_strings(['', '']) == 0
assert count_distinct_characters_across_strings(['Aa', 'aA']) == 1