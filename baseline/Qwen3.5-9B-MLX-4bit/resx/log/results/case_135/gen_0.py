
def can_arrange(arr):
    """Create a function which returns the largest index of an element which
    is not greater than or equal to the element immediately preceding it. If
    no such element exists then return -1. The given array will not contain
    duplicate values.

    Examples:
    can_arrange([1,2,4,3,5]) = 3
    can_arrange([1,2,3]) = -1
    """
    if not arr:
        return -1
    
    max_index = -1
    for i in range(1, len(arr)):
        if arr[i] < arr[i - 1]:
            max_index = i
    
    return max_index

def max_subarray_sum(arr):
    """Given a list of integers, find the maximum sum of a subarray where each subarray is defined by indices returned by the function 'can_arrange' applied to the original list. The subarray sum should be calculated for each valid subarray and the maximum sum should be returned. If no valid subarray exists, return 0.
    """
    # Get the largest index where the array is not sorted (descending)
    break_index = can_arrange(arr)
    
    if break_index == -1:
        # If the array is sorted in ascending order, the entire array is a valid subarray
        return sum(arr)
    
    # The valid subarrays are from index 0 to break_index and from break_index + 1 to the end
    subarray1 = arr[:break_index + 1]
    subarray2 = arr[break_index + 1:]
    
    sum1 = sum(subarray1)
    sum2 = sum(subarray2) if subarray2 else 0
    
    return max(sum1, sum2)


assert max_subarray_sum([1,2,4,3,5]) == 5
assert max_subarray_sum([1,2,3]) == 0
assert max_subarray_sum([5,4,3,2,1]) == 0
assert max_subarray_sum([10,20,30,20,10]) == 0