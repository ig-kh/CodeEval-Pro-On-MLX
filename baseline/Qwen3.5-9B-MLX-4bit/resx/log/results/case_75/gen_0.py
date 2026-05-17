
def is_multiply_prime(a):
    """Write a function that returns true if the given number is the multiplication of 3 prime numbers
    and false otherwise.
    Knowing that (a) is less then 100. 
    Example:
    is_multiply_prime(30) == True
    30 = 2 * 3 * 5
    """
    if a < 2:
        return False
    
    # Helper function to check if a number is prime
    def is_prime(num):
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
    
    # Find all prime factors of a
    factors = []
    temp = a
    for i in range(2, int(temp**0.5) + 1):
        while temp % i == 0:
            factors.append(i)
            temp //= i
    if temp > 1:
        factors.append(temp)
    
    # Check if there are exactly 3 prime factors
    return len(factors) == 3

def find_multiply_primes(numbers):
    """Write a function that takes a list of numbers and returns a list of numbers that are the multiplication of exactly 3 prime numbers, considering each number in the input list. The input list contains numbers less than 100."""
    result = []
    for num in numbers:
        if is_multiply_prime(num):
            result.append(num)
    return result


assert find_multiply_primes([30, 42, 60, 70, 84]) == [30, 42, 70]
assert find_multiply_primes([10, 15, 20, 25, 30]) == [20, 30]
assert find_multiply_primes([1, 2, 3, 4, 5]) == []
assert find_multiply_primes([2, 3, 5, 7, 11]) == []
assert find_multiply_primes([30, 42, 60, 70, 84, 90]) == [30, 42, 70]