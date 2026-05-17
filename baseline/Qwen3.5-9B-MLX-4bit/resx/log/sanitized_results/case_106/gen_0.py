
def f(n):
    """ Implement the function f that takes n as a parameter,
    and returns a list of size n, such that the value of the element at index i is the factorial of i if i is even
    or the sum of numbers from 1 to i otherwise.
    i starts from 1.
    the factorial of i is the multiplication of the numbers from 1 to i (1 * 2 * ... * i).
    Example:
    f(5) == [1, 2, 6, 24, 15]
    """
    result = []
    for i in range(1, n + 1):
        if i % 2 == 0:
            # Calculate factorial of i
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            result.append(fact)
        else:
            # Calculate sum of numbers from 1 to i
            result.append(i * (i + 1) // 2)
    return result

def g(lst):
    """
    Implement the function g that takes a list of integers as a parameter, and returns a list where each element at index i is the result of applying the function f to the element at index i in the input list. The function f should be called multiple times, once for each element in the input list. The function f is defined as follows: f(n) returns a list of size n, such that the value of the element at index i is the factorial of

assert g([1, 2, 3, 4, 5]) == [[1], [1, 2], [1, 2, 6], [1, 2, 6, 24], [1, 2, 6, 24, 15]]
assert g([0, 1, 2, 3, 4, 5]) == [[], [1], [1, 2], [1, 2, 6], [1, 2, 6, 24], [1, 2, 6, 24, 15]]
assert g([5, 4, 3, 2, 1]) == [[1, 2, 6, 24, 15], [1, 2, 6, 24], [1, 2, 6], [1, 2], [1]]
assert g([3, 3, 3]) == [[1, 2, 6], [1, 2, 6], [1, 2, 6]]