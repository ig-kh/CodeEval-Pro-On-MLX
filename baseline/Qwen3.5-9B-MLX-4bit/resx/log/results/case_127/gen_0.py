
def intersection(interval1, interval2):
    """You are given two intervals,
    where each interval is a pair of integers. For example, interval = (start, end) = (1, 2).
    The given intervals are closed which means that the interval (start, end)
    includes both start and end.
    For each given interval, it is assumed that its start is less or equal its end.
    Your task is to determine whether the length of intersection of these two 
    intervals is a prime number.
    Example, the intersection of the intervals (1, 3), (2, 4) is (2, 3)
    which its length is 1, which not a prime number.
    If the length of the intersection is a prime number, return "YES",
    otherwise, return "NO".
    If the two intervals don't intersect, return "NO".


    [input/output] samples:
    intersection((1, 2), (2, 3)) ==> "NO"
    intersection((-1, 1), (0, 4)) ==> "NO"
    intersection((-3, -1), (-5, 5)) ==> "YES"
    """
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # Calculate the intersection of the two intervals
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    
    # Check if the intervals intersect
    if start > end:
        return "NO"
    
    # Calculate the length of the intersection
    length = end - start + 1
    
    # Check if the length is a prime number
    if is_prime(length):
        return "YES"
    else:
        return "NO"

def find_prime_intersection_pair(intervals):
    """
    Given a list of intervals, determine if there exists a pair of intervals such that the length of their intersection is a prime number. If such a pair exists, return the indices of the intervals in the list. If no such pair exists, return an empty list. Note that the intervals are closed and each interval's start is less than or equal to its end.
    """
    result = []
    
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            # Call the intersection function to check if the length of the intersection is a prime number
            if intersection(intervals[i], intervals[j]) == "YES":
                result.append((i, j))
    
    return result


assert find_prime_intersection_pair([(1, 2), (2, 3)]) == []
assert find_prime_intersection_pair([(-1, 1), (0, 4)]) == []
assert find_prime_intersection_pair([(-3, -1), (-5, 5)]) == [0, 1]
assert find_prime_intersection_pair([(1, 3), (2, 4)]) == []
assert find_prime_intersection_pair([(1, 5), (3, 7), (6, 10)]) == [0, 1] 