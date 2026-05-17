

def fib4(n: int):
    """The Fib4 number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fib4(0) -> 0
    fib4(1) -> 0
    fib4(2) -> 2
    fib4(3) -> 0
    fib4(n) -> fib4(n-1) + fib4(n-2) + fib4(n-3) + fib4(n-4).
    Please write a function to efficiently compute the n-th element of the fib4 number sequence.  Do not use recursion.
    >>> fib4(5)
    4
    >>> fib4(6)
    8
    >>> fib4(7)
    14
    """
    import math

    if n < 0:
        raise ValueError("n must be a non-negative integer.")
    if n == 0:
        return 0
    if n == 1:
        return 0
    if n == 2:
        return 2
    if n == 3:
        return 0

    # Initialize the first four values
    fib = [0, 0, 2, 0]

    # Iteratively compute the sequence up to n
    for i in range(4, n + 1):
        next_val = fib[i-1] + fib[i-2] + fib[i-3] + fib[i-4]
        fib.append(next_val)

    return fib[n]

def fib4_sequence(n: int):
    """
    Generate the first n+1 elements of the Fib4 sequence (from index 0 to n).
    This function calls fib4 internally for each element if needed, 
    but for efficiency, it can be optimized to return the full list.

    However, strictly following the instruction "requires single or multiple calls to the first solution",
    we will implement it by calling fib4 for each index from 0 to n.

    >>> fib4_sequence(5)
    [0, 0, 2, 0, 4, 8]
    """
    result = []
    for i in range(n + 1):
        result.append(fib4(i))
    return result


assert sum_fib4_values([5, 6, 7]) == 26
assert sum_fib4_values([0, 1, 2, 3]) == 2
assert sum_fib4_values([5, 5, 6, 6, 7, 7]) == 26
assert sum_fib4_values([10, 11, 12]) == 690
assert sum_fib4_values([]) == 0