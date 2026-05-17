

def string_sequence(n: int) -> str:
    """ Return a string containing space-delimited numbers starting from 0 upto n inclusive.
    >>> string_sequence(0)
    '0'
    >>> string_sequence(5)
    '0 1 2 3 4 5'
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    
    return ' '.join(str(i) for i in range(n + 1))


def concatenated_sequences(lst: List[int]) -> str:
    """ Given a list of integers, return a single string that concatenates the space-delimited sequences starting from 0 up to each integer in the list.
    """
    if not lst:
        return ""
    
    result = []
    for num in lst:
        if num < 0:
            raise ValueError("All integers in the list must be non-negative.")
        result.append(string_sequence(num))
    
    return ' '.join(result)


assert concatenated_sequences([0, 5, 3]) == '0 0 1 2 3 4 5 0 1 2 3'
assert concatenated_sequences([]) == ''
assert concatenated_sequences([1, 1, 1]) == '0 1 0 1 0 1'
assert concatenated_sequences([10]) == '0 1 2 3 4 5 6 7 8 9 10'