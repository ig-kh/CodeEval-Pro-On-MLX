
def specialFilter(nums):
    """Write a function that takes an array of numbers as input and returns 
    the number of elements in the array that are greater than 10 and both 
    first and last digits of a number are odd (1, 3, 5, 7, 9).
    For example:
    specialFilter([15, -73, 14, -15]) => 1 
    specialFilter([33, -2, -3, 45, 21, 109]) => 2
    """
    import re

    def is_special(num):
        if num <= 10:
            return False

        num = abs(num)
        first_digit = int(str(num)[0])
        last_digit = num % 10

        return first_digit % 2 == 1 and last_digit % 2 == 1

    count = 0
    for num in nums:
        if is_special(num):
            count += 1

    return count

def totalSpecialFilter(nums_list):
    """
    Given a list of lists of numbers, write a function that returns the total count of elements across all sublists that are greater than 10 and both first and odd (1, 3, 5, 7, 9).
    """
    total_count = 0
    for sublist in nums_list:
        total_count += specialFilter(sublist)

    return total_count


assert totalSpecialFilter([[15, -73, 14, -15], [33, -2, -3, 45, 21, 109]]) == 3
assert totalSpecialFilter([[10, 20, 30], [40, 50, 60]]) == 0
assert totalSpecialFilter([[11, 13, 15], [17, 19, 21]]) == 5
assert totalSpecialFilter([[12, 14, 16], [18, 20, 22]]) == 0
assert totalSpecialFilter([[111, 113, 115], [117, 119, 121]]) == 6