from typing import List


def sort_numbers(numbers: str) -> str:
    """ Input is a space-delimited string of numberals from 'zero' to 'nine'.
    Valid choices are 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight' and 'nine'.
    Return the string with numbers sorted from smallest to largest
    >>> sort_numbers('three one five')
    'one three five'
    """
    number_map = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
       

@@ Response


assert sort_and_concatenate(['three one five', 'two four six']) == 'one three five two four six'
assert sort_and_concatenate(['nine eight seven', 'one two three']) == 'seven eight nine one two three'
assert sort_and_concatenate(['zero zero', 'invalid numeral']) == 'zero zero '
assert sort_and_concatenate(['', 'one two']) == ' one two'