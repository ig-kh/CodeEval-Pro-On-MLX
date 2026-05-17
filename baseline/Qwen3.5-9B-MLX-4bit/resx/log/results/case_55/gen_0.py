

def fib(n: int):
    """Return n-th Fibonacci number.
    >>> fib(10)
    55
    >>> fib(1)
    1
    >>> fib(8)
    21
    """
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def sum_of_fibs_except_self(lst: List[int]):
    """
    Given a list of integers, return a list where each element at index i is the sum of the Fibonacci numbers of all elements in the original list except the element at index i. For example, if the input list is [1, 

assert sum_of_fibs_except_self([1, 2, 3]) == [3, 3, 2]
assert sum_of_fibs_except_self([0, 1, 1, 2]) == [3, 2, 2, 2]
assert sum_of_fibs_except_self([5, 5, 5]) == [10, 10, 10]
assert sum_of_fibs_except_self([10, 20, 30]) == [838805, 832095, 6820]