
def get_odd_collatz(n):
    """
    Given a positive integer n, return a sorted list that has the odd numbers in collatz sequence.

    The Collatz conjecture is a conjecture in mathematics that concerns a sequence defined
    as follows: start with any positive integer n. Then each term is obtained from the 
    previous term as follows: if the previous term is even, the next term is one half of 
    the previous term. If the previous term is odd, the next term is 3 times the previous
    term plus 1. The conjecture is that no matter what value of n, the sequence will always reach 1.

    Note: 
        1. Collatz(1) is [1].
        2. returned list sorted in increasing order.

    For example:
    get_odd_collatz(5) returns [1, 5] # The collatz sequence for 5 is [5, 16, 8, 4, 2, 1], so the odd numbers are only 1, and 5.
    """
    if n == 1:
        return [1]
    
    odd_numbers = []
    current = n
    
    while current > 1:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1
        
        if current % 2 != 0:
            odd_numbers.append(current)
            
    return sorted(odd_numbers)

def get_all_odd_collatz(numbers):
    """
    Given a list of positive integers, return a sorted list of all odd numbers that appear in the Collatz sequences of these integers. The Collatz sequence for each integer should be computed independently.
    """
    all_odd_numbers = []
    
    for num in numbers:
        # Call the get_odd_collatz function for each number
        odd_nums = get_odd_collatz(num)
        all_odd_numbers.extend(odd_nums)
    
    return sorted(all_odd_numbers)


assert get_all_odd_collatz([5, 6]) == [1, 3, 5]
assert get_all_odd_collatz([7, 8]) == [1, 5, 7, 11, 13, 17]
assert get_all_odd_collatz([1, 2, 3]) == [1, 3, 5]
assert get_all_odd_collatz([10, 11]) == [1, 5, 11, 13, 17]
assert get_all_odd_collatz([13, 14]) == [1, 5, 7, 11, 13, 17]