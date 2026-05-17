
def count_up_to(n):
    """Implement a function that takes an non-negative integer and returns an array of the first n
    integers that are prime numbers and less than n.
    for example:
    count_up_to(5) => [2,3]
    count_up_to(11) => [2,3,5,7]
    count_up_to(0) => []
    count_up_to(20) => [2,3,5,7,11,13,17,19]
    count_up_to(1) => []
    count_up_to(18) => [2,3,5,7,11,13,17]
    """
    if n <= 1:
        return []
    
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    
    return primes

def generate_prime_dict(numbers):
    """
    Implement a function that takes a list of non-negative integers and returns a dictionary where each key is a unique integer from the list and the value is an array of the first n integers that are prime numbers and less than the key. For example, if the input list is [5, 11], the output should be {5: [2, 3], 11: [2, 3, 5, 7]}.
    """
    result = {}
    
    for num in numbers:
        if num > 1:
            result[num] = count_up_to(num)
    
    return result


assert generate_prime_dict([5, 11]) == {5: [2, 3], 11: [2, 3, 5, 7]}
assert generate_prime_dict([0, 1, 20]) == {0: [], 1: [], 20: [2, 3, 5, 7, 11, 13, 17, 19]}
assert generate_prime_dict([18, 1]) == {18: [2, 3, 5, 7, 11, 13, 17], 1: []}
assert generate_prime_dict([10, 15, 2]) == {10: [2, 3, 5, 7], 15: [2, 3, 5, 7, 11, 13], 2: []}