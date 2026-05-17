
def add_elements(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return
    the sum of the elements with at most two digits from the first k elements of arr.

    Example:

        Input: arr = [111,21,3,4000,5,6,7,8,9], k = 4
        Output: 24 # sum of 21 + 3

    Constraints:
        1. 1 <= len(arr) <= 100
        2. 1 <= k <= len(arr)
    """
    total_sum = 0
    
    # Iterate through the first k elements of arr
    for i in range(k):
        if i < len(arr):
            num = arr[i]
            # Check if the number has at most two digits
            if num >= 0 and num < 100:
                total_sum += num
                
    return total_sum

def sum_of_two_digit_elements_in_subarrays(arr, k):
    """
    Given a non-empty array of integers arr and an integer k, return the sum of the elements with at most two digits from the first k elements of arr for each subarray of length k in arr. If the length of arr is not a multiple of k, the last subarray will contain the remaining elements. For example, if arr = [111,21,3,40

assert sum_of_two_digit_elements_in_subarrays([111,21,3,4000,5,6,7,8,9], 4) == 59
assert sum_of_two_digit_elements_in_subarrays([1,2,3,4,5,6,7,8,9], 3) == 45
assert sum_of_two_digit_elements_in_subarrays([1000,2000,3000,4000,5000], 2) == 0