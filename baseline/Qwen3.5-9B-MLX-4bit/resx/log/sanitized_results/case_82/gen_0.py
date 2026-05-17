
def prime_length(string):
    """Write a function that takes a string and returns True if the string
    length is a prime number or False otherwise
    Examples
    prime_length('Hello') == True
    prime_length('abcdcba') == True
    prime_length('kittens') == True
    prime_length('orange') == False
    """
    def is_prime(n):
        if n <= 1:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False

        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False

        return True

    length = len(string)
    return is_prime(length)

def prime_length_count_sum(strings):
    """Given a list of strings, write a function that returns the count of strings with prime length and the sum of their lengths. The function should return a tuple where the first element is the count and the second element is the sum of the lengths."""
    count = 0
    total_sum = 0

    for s in strings:
        if prime_length(s):
            count += 1
            total_sum += len(s)

    return (count, total_sum)


assert prime_length_count_sum(['Hello', 'abcdcba', 'kittens', 'orange']) == (3, 19)
assert prime_length_count_sum(['a', 'bb', 'ccc', 'dddd']) == (2, 5)
assert prime_length_count_sum(['racecar', 'level', 'stats']) == (3, 17)