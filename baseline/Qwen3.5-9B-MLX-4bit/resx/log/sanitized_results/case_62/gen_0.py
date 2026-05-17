

def derivative(xs: list):
    """ xs represent coefficients of a polynomial.
    xs[0] + xs[1] * x + xs[2] * x^2 + ....
     Return derivative of this polynomial in the same form.
    >>> derivative([3, 1, 2, 4, 5])
    [1, 4, 12, 20]
    >>> derivative([1, 2, 3])
    [2, 6]
    """
    if not xs:
        return []
    return [i * xs[i] for i in range(1, len(xs))]

def second_derivative(polynomials: list) -> list:
    """second_derivative is a function that takes a list of polynomials and returns the second derivative of each polynomial.
    >>> second_derivative([[3, 1, 2, 4, 5], [1, 2, 3]])
    [[4, 12, 20], [6]]
    >>> second_derivative([[1, 2, 3]])
    [[2, 6]]
    >>> second_derivative([])
    []
    """
    return [derivative(derivative(p)) for p in polynomials]

assert second_derivative([[3, 1, 2, 4, 5], [1, 2, 3]]) == [[4, 24, 60], [6]]
assert second_derivative([[1, 0, 1], [2, 0, 2]]) == [[2], [4]]
assert second_derivative([[5], [7]]) == [[], []]