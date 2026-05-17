
def minSubArraySum(nums):
    """
    Given an array of integers nums, find the minimum sum of any non-empty sub-array
    of nums.
    Example
    minSubArraySum([2, 3, 4, 1, 2, 4]) == 1
    minSubArraySum([-1, -2, -3]) == -6
    """
    import re

    if not nums:
        return 0

    min_sum = float('inf')
    current_sum = 0

    for num in nums:
        current_sum += num
        if current_sum < min_sum:
            min_sum = current_sum

        if current_sum <= 0:
            current_sum = 0

    return min_sum

def maxSubArraySumGreaterThanT(nums, T):
    """
    Given an array of integers nums, find the maximum sum of any non-empty sub-array of nums that is greater than a given threshold T. If no such sub-array exists, return -1. To solve this problem, you need to first find the minimum sum of any non-empty sub-array of nums, then use this information to determine the maximum sum greater than T.
    """
    # Find the minimum sub-array sum using the first function
    min_sum = minSubArraySum(nums)

    # Calculate the maximum sub-array sum using Kadane's algorithm
    max_sum = 0
    current_sum = 0

    for num in nums:
        current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum

        if current_sum <= 0:
            current_sum = 0

    # Check if the maximum sum is greater than T
    if max_sum > T:
        return max_sum
    else:
        return -1


assert maxSubArraySumGreaterThanT([2, 3, 4, 1, 2, 4], 5) == 16
assert maxSubArraySumGreaterThanT([-1, -2, -3], -5) == -1
assert maxSubArraySumGreaterThanT([1, 2, 3, 4, 5], 10) == 15
assert maxSubArraySumGreaterThanT([1, -1, 1, -1, 1], 2) == -1