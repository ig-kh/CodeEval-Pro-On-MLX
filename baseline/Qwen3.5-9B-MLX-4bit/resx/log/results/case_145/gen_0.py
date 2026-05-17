
def order_by_points(nums):
    """
    Write a function which sorts the given list of integers
    in ascending order according to the sum of their digits.
    Note: if there are several items with similar sum of their digits,
    order them based on their index in original list.

    For example:
    >>> order_by_points([1, 11, -1, -11, -12]) == [-1, -11, 1, -12, 11]
    >>> order_by_points([]) == []
    """
    def sum_digits(n):
        n = abs(n)
        return sum(int(digit) for digit in str(n))
    
    def get_sort_key(num):
        return (sum_digits(num), nums.index(num))
    
    return sorted(nums, key=get_sort_key)

def complex_order_by_points(nums_list):
    """
    Given a list of lists of integers, write a function that sorts each sublist according to the sum of their digits and then sorts the main list based on the sum of the digits of the first element of each sublist. If two sublists have the same sum for their first elements, sort them based on their original index in the main list.
    """
    def sum_digits(n):
        n = abs(n)
        return sum(int(digit) for digit in str(n))
    
    def get_sort_key(num):
        return (sum_digits(num), nums_list.index(num))
    
    result = []
    for i, sublist in enumerate(nums_list):
        sorted_sublist = sorted(sublist, key=get_sort_key)
        result.append(sorted_sublist)
    
    return result


assert complex_order_by_points([[1, 11], [-1, -11], [-12]]) == [[1, 11], [-1, -11], [-12]]
assert complex_order_by_points([[100, 200], [9, 99], [8, 88]]) == [[100, 200], [8, 88], [9, 99]]
assert complex_order_by_points([[1], [11], [111]]) == [[1], [11], [111]]
assert complex_order_by_points([[], [1, 2], [3, 4]]) == [[], [1, 2], [3, 4]]
assert complex_order_by_points([[1, 11], [-1, -11], [-12], [2, 22]]) == [[1, 11], [-1, -11], [2, 22], [-12]]