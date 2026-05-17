
def tri(n):
    """Everyone knows Fibonacci sequence, it was studied deeply by mathematicians in 
    the last couple centuries. However, what people don't know is Tribonacci sequence.
    Tribonacci sequence is defined by the recurrence:
    tri(1) = 3
    tri(n) = 1 + n / 2, if n is even.
    tri(n) =  tri(n - 1) + tri(n - 2) + tri(n + 1), if n is odd.
    For example:
    tri(2) = 1 + (2 / 2) = 2
    tri(4) = 3
    tri(3) = tri(2) + tri(1) + tri(4)
           = 2 + 3 + 3 = 8 
    You are given a non-negative integer number n, you have to a return a list of the 
    first n + 1 numbers of the Tribonacci sequence.
    Examples:
    tri(3) = [1, 3, 2, 8]
    """
    # Memoization dictionary to store computed values
    memo = {0: 1, 1: 3}
    
    def compute(k):
        if k in memo:
            return memo[k]
        
        if k % 2 == 0:
            result = 1 + k // 2
        else:
            # For odd k, we need tri(k-1), tri(k-2), and tri(k+1)
            # Since k+1 is even, we can compute it directly
            result = compute(k - 1) + compute(k - 2) + (1 + (k + 1) // 2)
        
        memo[k] = result
        return result
    
    # Compute all values up to n
    for i in range(2, n + 1):
        compute(i)
    
    return [memo[i] for i in range(n + 1)]

def multi_tri(n):
    """Given a non-negative integer number n, you have to return a list of lists where 
    each sublist is the Tribonacci sequence for the numbers 0 through n.
    
    For example, if n is 3, you should return a list of lists where:
    - The first sublist is the sequence for 0 (just [1])
    - The second sublist is the sequence for 0 through 1 ([1, 3])
    - The third sublist is the sequence for 0 through 2 ([1, 3, 2])
    - The fourth sublist is the sequence for 0 through 3 ([1, 3, 2, 8])
    
    Examples:
    multi_tri(0) = [[1]]
    multi_tri(3) = [[1], [1, 3], [1, 3, 2], [1, 3, 2, 8]]
    """
    # Use the tri function to get the full sequence up to n
    full_sequence = tri(n)
    
    # Build the list of lists by taking prefixes of the full sequence
    result = []
    for i in range(n + 1):
        result.append(full_sequence[:i + 1])
    
    return result


assert multi_tri(3) == [[1], [1, 3], [1, 3, 2], [1, 3, 2, 8]]
assert multi_tri(0) == [[1]]
assert multi_tri(1) == [[1], [1, 3]]