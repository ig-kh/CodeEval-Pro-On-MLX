
def total_match(lst1, lst2):
    '''
    Write a function that accepts two lists of strings and returns the list that has 
    total number of chars in the all strings of the list less than the other list.

    if the two lists have the same number of chars, return the first list.

    Examples
    total_match([], []) ➞ []
    total_match(['hi', 'admin'], ['hI', 'Hi']) ➞ ['hI', 'Hi']
    total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project']) ➞ ['hi', 'admin']
    total_match(['hi', 'admin'], ['hI', 'hi', 'hi']) ➞ ['hI', 'hi', 'hi']
    total_match(['4'], ['1', '2', '3', '4', '5']) ➞ ['4']
    '''
    import re

    def total_chars(lst):
        return sum(len(s) for s in lst)

    if not lst1 and not lst2:
        return []

    if total_chars(lst1) < total_chars(lst2):
        return lst1
    else:
        return lst2

def smallest_total_match(list_of_lists):
    '''
    Given a list of lists of strings, return the list of strings that has the smallest total number of characters across all its sublists. If there are multiple lists with the same smallest total number of characters, return the first one encountered.
    '''
    if not list_of_lists:
        return []

    # Calculate the total number of characters for each sublist
    totals = [total_chars(lst) for lst in list_of_lists]

    # Find the index of the sublist with the smallest total number of characters
    min_index = totals.index(min(totals))

    # Return the sublist with the smallest total number of characters
    return list_of_lists[min_index]


assert smallest_total_match([[], []]) == []
assert smallest_total_match([['hi', 'admin'], ['hI', 'Hi']]) == ['hI', 'Hi']
assert smallest_total_match([['hi', 'admin'], ['hi', 'hi', 'admin', 'project']]) == ['hi', 'admin']
assert smallest_total_match([['hi', 'admin'], ['hI', 'hi', 'hi']]) == ['hI', 'hi', 'hi']
assert smallest_total_match([['4'], ['1', '2', '3', '4', '5']]) == ['4']