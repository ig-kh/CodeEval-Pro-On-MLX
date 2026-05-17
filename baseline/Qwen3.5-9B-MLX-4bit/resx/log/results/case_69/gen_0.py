
def search(lst):
    '''
    You are given a non-empty list of positive integers. Return the greatest integer that is greater than 
    zero, and has a frequency greater than or equal to the value of the integer itself. 
    The frequency of an integer is the number of times it appears in the list.
    If no such a value exist, return -1.
    Examples:
        search([4, 1, 2, 2, 3, 1]) == 2
        search([1, 2, 2, 3, 3, 3, 4, 4, 4]) == 3
        search([5, 5, 4, 4, 4]) == -1
    '''
    from collections import Counter
    counts = Counter(lst)
    result = -1
    for num, freq in counts.items():
        if num > 0 and freq >= num:
            result = max(result, num)
    return result

def new_search(lst_of_lsts):
    """new_search is a function that takes a list of lists of positive integers and returns the sum of the results of applying search to each sublist.
    >>> new_search([[4, 1, 2, 2, 3, 1], [1, 2, 2, 3, 3, 3, 4, 4, 4], [5, 5, 4, 4, 4]])
    1
    >>> new_search([[1, 2, 2, 3, 3, 3, 4, 4, 4], [5, 5, 4, 4, 4]])
    2
    >>> new_search([])
    0
    """
    return sum(search(sublist) for sublist in lst_of_lsts)

assert new_search([[4, 1, 2, 2, 3, 1], [1, 2, 2, 3, 3, 3, 4, 4, 4], [5, 5, 4, 4, 4]]) == 4
assert new_search([[1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]]) == 6
assert new_search([[1], [2], [3]]) == -1
assert new_search([[1, 2, 2, 3, 3, 3, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6, 6]]) == 15