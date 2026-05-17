
def sort_array(arr):
    """
    In this Kata, you have to sort an array of non-negative integers according to
    number of ones in their binary representation in ascending order.
    For similar number of ones, sort based on decimal value.

    It must be implemented like this:
    >>> sort_array([1, 5, 2, 3, 4]) == [1, 2, 3, 4, 5]
    >>> sort_array([-2, -3, -4, -5, -6]) == [-6, -5, -4, -3, -2]
    >>> sort_array([1, 0, 2, 3, 4]) [0, 1, 2, 3, 4]
    """
    def count_ones(n):
        return bin(n).count('1')
    
    # Sort the array based on the number of ones in binary representation, then by decimal value
    return sorted(arr, key=lambda x: (count_ones(x), x))

def new_sort_array(arr):
    """Given a list of lists, each containing non-negative integers, sort each sublist according to the number of ones in their binary representation in ascending order. For sublists with similar number of ones, sort based on the decimal value. Finally, sort the main list based on the sum of the number of ones in the binary representation of all elements in each sublist."""
    # Sort each sublist using the sort_array function
    sorted_sublists = [sort_array(sublist) for sublist in arr]
    
    # Calculate the total number of ones for each sublist
    def count_ones(n):
        return bin(n).count('1')
    
    # Sort the main list based on the sum of ones in each sublist
    sorted_sublists.sort(key=lambda x: sum(count_ones(num) for num in x))
    
    return sorted_sublists


assert new_sort_array([[1, 5, 2, 3, 4], [7, 8, 9], [10, 11, 12]]) == [[8, 9, 7], [1, 2, 4, 3, 5], [10, 12, 11]]
assert new_sort_array([[15, 16, 17], [18, 19, 20], [21, 22, 23]]) == [[16, 17, 15], [18, 20, 19], [21, 22, 23]]
assert new_sort_array([[1, 0, 2, 3, 4], [5, 6, 7], [8, 9, 10]]) == [[0, 1, 2, 4, 3], [8, 9, 10], [5, 6, 7]]