
def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.

    Example 1:

        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]

    Example 2:

        Input: arr = [4, -4, 4], k = 2
        Output: [4, 4]

    Example 3:

        Input: arr = [-3, 2, 1, 2, -1, -2, 1], k = 1
        Output: [2]

    Note:
        1. The length of the array will be in the range of [1, 1000].
        2. The elements in the array will be in the range of [-1000, 1000].
        3. 0 <= k <= len(arr)
    """
    if k == 0:
        return []
    
    # Sort the array in descending order and take the top k elements
    sorted_arr = sorted(arr, reverse=True)
    
    # Take the top k elements
    top_k = sorted_arr[:k]
    
    # Sort the result in ascending order
    return sorted(top_k)

def maximum_for_multiple_arrays(arrays, ks):
    """
    Given a list of arrays, each containing integers, and a list of positive integers representing the values of k for each corresponding array, return a list of sorted lists where each sorted list contains the maximum k numbers from the corresponding array. If an array has fewer elements than k, return the sorted list of all elements in that array.
    """
    result = []
    
    for arr, k in zip(arrays, ks):
        # Call the maximum function to get the top k elements
        top_k = maximum(arr, k)
        result.append(top_k)
    
    return result


assert maximum_for_multiple_arrays([[-3, -4, 5], [4, -4, 4], [-3, 2, 1, 2, -1, -2, 1]], [3, 2, 1]) == [[-4, -3, 5], [4, 4], [2]]