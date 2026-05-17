

def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
        if is_prime(fibs[-1]):
            fibs.append(fibs[-1])
    return fibs[n - 2]

def prime_fib_sum(n: int):
    """
    Write a function `prime_fib_sum(n: int)` that returns the sum of the first `n` prime Fibonacci numbers. For example, if `n` is 

assert prime_fib_sum(1) == 2
assert prime_fib_sum(2) == 5
assert prime_fib_sum(3) == 10
assert prime_fib_sum(4) == 23
assert prime_fib_sum(5) == 112