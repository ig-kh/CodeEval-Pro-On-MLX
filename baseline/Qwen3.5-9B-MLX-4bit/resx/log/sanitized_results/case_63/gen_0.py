

def fibfib(n: int):
    """The FibFib number sequence is a sequence similar to the Fibbonacci sequnece that's defined as follows:
    fibfib(0) == 0
    fibfib(1) == 0
    fibfib(2) == 1
    fibfib(n) == fibfib(n-1) + fibfib(n-2) + fibfib(n-3).
    Please write a function to efficiently compute the n-th element of the fibfib number sequence.
    >>> fibfib(1)
    0
    >>> fibfib(5)
    4
    >>> fibfib(8)
    24
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return 0
    if n == 2:
        return 1

    fib = [0] * (n + 1)
    fib[2] = 1
    for i in range(3, n + 1):
        fib[i] = fib[i-1] + fib[i-2] + fib[i-3]
    return fib[n]

def fibfib_sum(lst: list):
    """fibfib_sum is a function that returns the sum of the FibFib numbers at the positions given in the list.
    >>> fibfib_sum([1, 5, 8])
    24
    """
    return sum(fibfib(x) for x in lst)


assert sum_fibfib_positions([1, 5, 8]) == 28
assert sum_fibfib_positions([0, 2, 4]) == 3
assert sum_fibfib_positions([3, 6, 9]) == 52
assert sum_fibfib_positions([10, 11, 12]) == 504