

def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """
    return len(string)


def total_length(strings: list) -> int:
    """ Given a list of strings, write a function to return the total length of all strings combined. If the strlen function is used to calculate the length of each string in the list, and then sum them up.
    >>> total_length([])
    0
    >>> total_length(['a', 'bc', 'def'])
    6
    """
    return sum(strlen(s) for s in strings)


assert total_length(['hello', 'world']) == 10
assert total_length(['python', 'programming']) == 17
assert total_length([]) == 0
assert total_length(['a', 'bc', 'def']) == 6