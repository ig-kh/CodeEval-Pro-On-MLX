from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """ Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
    if not strings:
        return None
    
    max_len = 0
    result = None
    
    for s in strings:
        if len(s) > max_len:
            max_len = len(s)
            result = s
    
    return result


def find_longest_strings(list_of_lists: List[List[str]]) -> List[str]:
    """ Given a sublist is empty, return 'EMPTY' for that sublist. If there are multiple strings of the same length in a sublist, return the first one. Return a list of these longest strings.
    """
    result = []
    for sublist in list_of_lists:
        if not sublist:
            result.append('EMPTY')
        else:
            longest_str = longest(sublist)
            if longest_str is None:
                result.append('EMPTY')
            else:
                result.append(longest_str)
    
    return result


assert find_longest_strings([['a', 'b', 'c'], ['d', 'ee', 'fff'], []]) == ['a', 'fff', 'EMPTY']
assert find_longest_strings([['a', 'bb', 'ccc'], ['d', 'ee', 'fff'], ['g', 'hh', 'iii']]) == ['ccc', 'fff', 'iii']
assert find_longest_strings([[], [], []]) == ['EMPTY', 'EMPTY', 'EMPTY']
assert find_longest_strings([['a'], ['bb'], ['ccc']]) == ['a', 'bb', 'ccc']