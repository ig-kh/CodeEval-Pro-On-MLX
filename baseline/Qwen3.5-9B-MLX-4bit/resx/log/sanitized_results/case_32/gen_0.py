import math


def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees
    a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    import re

    # Binary search for the root of the polynomial
    low, high = -1000.0, 1000.0
    while poly(xs, low) * poly(xs, high) > 0:
        mid = (low + high) / 2
        if poly(xs, mid) == 0:
            return mid
        low = mid if poly(xs, low) * poly(xs, mid) > 0 else high
    return (low + high) / 2

def find_common_zero(polys: list):
    """Given a list of polynomials, each represented by their coefficients, find the common zero point for all polynomials if it exists. If no common zero point exists, return None. Note that each polynomial is guaranteed to have an

assert find_common_zero([[1, 2], [-2, 4]]) == None
assert find_common_zero([[-6, 11, -6, 1], [1, -3, 2]]) == None
assert find_common_zero([[1, 2], [1, 3]]) == None
assert find_common_zero([]) == None
