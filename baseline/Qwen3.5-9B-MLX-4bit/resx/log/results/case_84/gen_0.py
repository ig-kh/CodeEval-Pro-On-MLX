
def solve(N):
    """Given a positive integer N, return the total sum of its digits in binary.
    
    Example
        For N = 1000, the sum of digits will be 1 the output should be "1".
        For N = 150, the sum of digits will be 6 the output should be "110".
        For N = 147, the sum of digits will be 12 the output should be "1100".
    
    Variables:
        @N integer
             Constraints: 0 ≤ N ≤ 10000.
    Output:
         a string of binary number
    """
    if N < 0:
        return "0"
    
    digit_sum = 0
    while N > 0:
        digit_sum += N % 10
        N //= 10
    
    return bin(digit_sum)[2:]

def new_solve(numbers):
    """Given a list of positive integers, return the concatenated binary representation of the sum of their digit sums. Each integer in the list should be processed individually to find its digit sum, and then the binary representation of the total sum of these digit sums should be returned."""
    total_digit_sum = 0
    
    for num in numbers:
        if num < 0:
            continue
        
        temp = num
        digit_sum = 0
        while temp > 0:
            digit_sum += temp % 10
            temp //= 10
        
        total_digit_sum += digit_sum
    
    return bin(total_digit_sum)[2:]


assert new_solve([1000, 150, 147]) == '10010111011'
assert new_solve([123, 456, 789]) == '10111110111101'
assert new_solve([10, 100, 1000]) == '11'
assert new_solve([1, 2, 3]) == '10110'
assert new_solve([9999, 1]) == '11000011100000101'