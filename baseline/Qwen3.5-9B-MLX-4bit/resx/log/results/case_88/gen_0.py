
def sort_array(array):
    """
    Given an array of non-negative integers, return a copy of the given array after sorting,
    you will sort the given array in ascending order if the sum( first index value, last index value) is odd,
    or sort it in descending order if the sum( first index value, last index value) is even.

    Note:
    * don't change the given array.

    Examples:
    * sort_array([]) => []
    * sort_array([5]) => [5]
    * sort_array([2, 4, 3, 0, 1, 5]) => [0, 1, 2, 3, 4, 5]
    * sort_array([2, 4, 3, 0, 1, 5, 6]) => [6, 5, 4, 3, 2, 1, 0]
    """
    if not array:
        return []
    
    first_val = array[0]
    last_val = array[-1]
    
    if (first_val + last_val) % 2 == 1:
        return sorted(array)
    else:
        return sorted(array, reverse=True)

def combine_and_sort_arrays(arrays):
    """Given a list of arrays, each containing non-negative integers, return a single sorted array that combines all the given arrays. The combined array should be sorted in ascending order if the total number of elements across all arrays is odd, or in descending order if the total number of elements is even. To achieve this, you

assert combine_and_sort_arrays([[2, 4, 3, 0, 1, 5], [6, 7, 8]]) == [8, 7, 6, 5, 4, 3, 2, 1, 0]
assert combine_and_sort_arrays([[2, 4, 3, 0, 1, 5], [6, 7]]) == [0, 1, 2, 3, 4, 5, 6, 7]
assert combine_and_sort_arrays([[5], [3], [1]]) == [5, 3, 1]
assert combine_and_sort_arrays([[], [2, 4, 3, 0, 1, 5, 6]]) == [6, 5, 4, 3, 2, 1, 0]