
def sorted_list_sum(lst):
    """Write a function that accepts a list of strings as a parameter,
    deletes the strings that have odd lengths from it,
    and returns the resulted list with a sorted order,
    The list is always a list of strings and never an array of numbers,
    and it may contain duplicates.
    The order of the list should be ascending by length of each word, and you
    should return the list sorted by that rule.
    If two words have the same length, sort the list alphabetically.
    The function should return a list of strings in sorted order.
    You may assume that all words will have the same length.
    For example:
    assert list_sort(["aa", "a", "aaa"]) => ["aa"]
    assert list_sort(["ab", "a", "aaa", "cd"]) => ["ab", "cd"]
    """
    import math
    import re

    lst = [s for s in lst if len(s) % 2 == 0]
    return sorted(lst, key=lambda x: (len(x), x))

def merge_sorted_lists(list_of_lists):
    """
    Given a list of lists of strings, where each list of strings is sorted by the rules defined in the raw question, write a function that merges all these lists into a single sorted list. The merged list should also follow the same sorting rules: sorted by length of each word in ascending order, and alphabetically for words of the same length. The function should return the merged list of strings.
    """
    merged_list = []
    for sublist in list_of_lists:
        merged_list.extend(sublist)

    return sorted(merged_list, key=lambda x: (len(x), x))


assert merge_sorted_lists([['aa', 'a', 'aaa'], ['ab', 'a', 'aaa', 'cd']]) == ['aa', 'ab', 'cd']
assert merge_sorted_lists([['bb', 'b', 'bbb'], ['cc', 'c', 'ccc', 'dd']]) == ['bb', 'cc', 'dd']
assert merge_sorted_lists([['aa', 'a', 'aaa'], ['aa', 'a', 'aaa']]) == ['aa', 'aa']
assert merge_sorted_lists([['ab', 'a', 'aaa', 'cd'], ['bb', 'b', 'bbb', 'cc']]) == ['ab', 'bb', 'cc', 'cd']