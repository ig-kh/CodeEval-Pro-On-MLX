
def move_one_ball(arr):
    """We have an array 'arr' of N integers arr[1], arr[2], ..., arr[N].The
    numbers in the array will be randomly ordered. Your task is to determine if
    it is possible to get an array sorted in non-decreasing order by performing 
    the following operation on the given array:
        You are allowed to perform right shift operation any number of times.
    
    One right shift operation means shifting all elements of the array by one
    position in the right direction. The last element of the array will be moved to
    the starting position in the array i.e. 0th index. 

    If it is possible to obtain the sorted array by performing the above operation
    then return True else return False.
    If the given array is empty then return True.

    Note: The given list is guaranteed to have unique elements.

    For Example:
    
    move_one_ball([3, 4, 5, 1, 2])==>True
    Explanation: By performin 2 right shift operations, non-decreasing order can
                 be achieved for the given array.
    move_one_ball([3, 5, 4, 1, 2])==>False
    Explanation:It is not possible to get non-decreasing order for the given
                array by performing any number of right shift operations.
                
    """
    if not arr:
        return True

    # Find the number of descents (where arr[i] > arr[i+1])
    descents = 0
    n = len(arr)

    for i in range(n):
        if arr[i] > arr[(i + 1) % n]:
            descents += 1

    # If there is at most one descent, it's possible to sort by right shifts
    return descents <= 1

def can_sort_all_arrays(arrays):
    """
    Given a list of arrays, each containing unique integers, determine if it is possible to sort each array in non-decreasing order by performing right shift operations any number of times. Return a list of boolean values indicating whether each array can be sorted.
    """
    result = []
    for arr in arrays:
        result.append(move_one_ball(arr))
    return result


assert can_sort_all_arrays([[3, 4, 5, 1, 2], [3, 5, 4, 1, 2]]) == [True, False]
assert can_sort_all_arrays([[1, 2, 3, 4, 5], [5, 1, 2, 3, 4]]) == [True, True]
assert can_sort_all_arrays([[2, 1], [1, 2]]) == [True, True]
assert can_sort_all_arrays([[], [1]]) == [True, True]