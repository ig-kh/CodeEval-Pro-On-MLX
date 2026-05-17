
def simplify(x, n):
    """Your task is to implement a function that will simplify the expression
    x * n. The function returns True if x * n evaluates to a whole number and False
    otherwise. Both x and n, are string representation of a fraction, and have the following format,
    <numerator>/<denominator> where both numerator and denominator are positive whole numbers.

    You can assume that x, and n are valid fractions, and do not have zero as denominator.

    simplify("1/5", "5/1") = True
    simplify("1/6", "2/1") = False
    simplify("7/10", "10/2") = False
    """
    def parse_fraction(frac):
        numerator, denominator = frac.split('/')
        return int(numerator), int(denominator)
    
    num_x, den_x = parse_fraction(x)
    num_n, den_n = parse_fraction(n)
    
    product_num = num_x * num_n
    product_den = den_x * den_n
    
    return product_num % product_den == 0

def find_whole_number_products(fractions):
    """
    Given a list of fractions represented as strings in the format <numerator>/<denominator>, determine if the product of any two fractions in the list simplifies to a whole number. Return a list of tuples where each tuple contains the indices of the two fractions whose product simplifies to a whole number.
    """
    result = []
    for i in range(len(fractions)):
        for j in range(i + 1, len(fractions)):
            if simplify(fractions[i], fractions[j]):
                result.append((i, j))
    
    return result


assert find_whole_number_products(['1/5', '5/1', '2/3']) == [(0, 1)]
assert find_whole_number_products(['1/6', '2/1', '3/2']) == [(1, 2)]
assert find_whole_number_products(['7/10', '10/2', '1/1']) == [(1, 2)]