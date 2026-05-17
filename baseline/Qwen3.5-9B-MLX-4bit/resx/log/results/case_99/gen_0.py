
def closest_integer(value):
    '''
    Create a function that takes a value (string) representing a number
    and returns the closest integer to it. If the number is equidistant
    from two integers, round it away from zero.

    Examples
    >>> closest_integer("10")
    10
    >>> closest_integer("15.3")
    15

    Note:
    Rounding away from zero means that if the given number is equidistant
    from two integers, the one you should return is the one that is the
    farthest from zero. For example closest_integer("14.5") should
    return 15 and closest_integer("-14.5") should return -15.
    '''
    num = float(value)
    
    # Calculate the distance to the nearest integers
    lower = int(num)
    upper = lower + 1
    
    if num < 0:
        upper = lower - 1
    
    distance_to_lower = abs(num - lower)
    distance_to_upper = abs(num - upper)
    
    if distance_to_lower < distance_to_upper:
        return lower
    elif distance_to_upper < distance_to_lower:
        return upper
    else:
        # Equidistant, round away from zero
        if num > 0:
            return upper
        else:
            return lower

def closest_integers(values):
    """
    Create a function that takes a list of strings, each representing a number, and returns a list of the closest integers to each number. If a number is equidistant from two integers, round it away from zero. The function should handle multiple calls to the 'closest_integer' function to achieve this.
    """
    result = []
    
    for value in values:
        result.append(closest_integer(value))
    
    return result


assert closest_integers(['10', '15.3']) == [10, 15]
assert closest_integers(['14.5', '-14.5']) == [15, -15]
assert closest_integers(['2.49', '2.50', '2.51']) == [2, 3, 3]
assert closest_integers(['-2.49', '-2.50', '-2.51']) == [-2, -3, -3]
assert closest_integers(['0', '0.5', '-0.5']) == [0, 1, -1]