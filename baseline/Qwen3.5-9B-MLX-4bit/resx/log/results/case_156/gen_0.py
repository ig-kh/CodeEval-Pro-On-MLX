
def int_to_mini_roman(number):
    """
    Given a positive integer, obtain its roman numeral equivalent as a string,
    and return it in lowercase.
    Restrictions: 1 <= num <= 1000

    Examples:
    >>> int_to_mini_roman(19) == 'xix'
    >>> int_to_mini_roman(152) == 'clii'
    >>> int_to_mini_roman(426) == 'cdxxvi'
    """
    roman_map = {
        1000: 'm',
        900: 'cm',
        500: 'd',
        400: 'cd',
        100: 'c',
        90: 'xc',
        50: 'l',
        10: 'x',
        9: 'ix',
        5: 'v',
        4: 'iv',
        1: 'i'
    }
    
    result = ''
    for value, symbol in sorted(roman_map.items(), reverse=True):
        while number >= value:
            result += symbol
            number -= value
    
    return result.lower()

def list_to_mini_roman(numbers):
    """Given a list of positive integers, obtain their roman numeral equivalents as a list of strings, each in lowercase. Restrictions: Each integer in the list should be within the range 1 to 

assert list_to_mini_roman([19, 152, 426]) == ['xix', 'clii', 'cdxxvi']
assert list_to_mini_roman([1, 1000, 500]) == ['i', 'm', 'd']
assert list_to_mini_roman([399, 888, 999]) == ['cccxcix', 'dccclxxxviii', 'cmxcix']