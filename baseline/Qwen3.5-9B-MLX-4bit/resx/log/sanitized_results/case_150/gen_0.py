
def x_or_y(n, x, y):
    """A simple program which should return the value of x if n is 
    a prime number and should return the value of y otherwise.

    Examples:
    for x_or_y(7, 34, 12) == 34
    for x_or_y(15, 8, 5) == 5
    
    """
    if n < 2:
        return y
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return y
    return x

def sum_x_or_y(numbers, x_values, zipped):
    """Given a list of integers, determine the sum of the values of x for all prime numbers in the list and the sum of the values of y values for all non-prime numbers in the
    The function should take a list of integers, a list of x values, and a list of y values as inputs. The length of the list of integers will always be equal to the length of the lists of x and y values.

    Examples:
    sum_x_or_y([7, 15], [34, 8], [12, 5]) -> 34 + 5 = 39
    sum_x_or_y([2, 3], [10, 20], [5, 5]) -> 10 + 20 = 30
    """
    total_sum = 0
    for i in range(len(numbers)):
        if zipped:
            total_sum += x_or_y(numbers[i], x_values[i], y_values[i])
        else:
            total_sum += x_or_y(numbers[i], x_values[i], y_values[i])
    return total_sum


assert sum_x_or_y([7, 15], [34, 8], [12, 5]) == (34, 5)
assert sum_x_or_y([2, 4, 6], [1, 2, 3], [4, 5, 6]) == (1, 11)
assert sum_x_or_y([11, 13, 17], [10, 20, 30], [5, 15, 25]) == (60, 0)
assert sum_x_or_y([1, 4, 6, 8], [1, 2, 3, 4], [5, 6, 7, 8]) == (0, 26)