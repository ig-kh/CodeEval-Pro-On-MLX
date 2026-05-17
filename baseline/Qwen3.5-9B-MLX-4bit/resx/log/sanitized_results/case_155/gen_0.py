
def even_odd_count(num):
    """Given an integer. return a tuple that has the number of even and odd digits respectively.

     Example:
        even_odd_count(-12) ==> (1, 1)
        even_odd_count(123) ==> (1, 2)
    """
    count_even = 0
    count_odd = 0

    num = abs(num)

    while num > 0:
        digit = num % 10
        if digit % 2 == 0:
            count_even += 1
        else:
            count_odd += 1
        num //= 10

    return (count_even, count_odd)

def total_even_odd_count(nums):
    """Given a list of integers, return a tuple where the first element is the total number of even digits across all integers in the list, and the second element is the total number of odd digits across all integers in the list.
    """
    total_even = 0
    total_odd = 0

    for num in nums:
        even, odd = even_odd_count(num)
        total_even += even
        total_odd += odd

    return (total_even, total_odd)


assert total_even_odd_count([-12, 123]) == (2, 3)
assert total_even_odd_count([0, 111, 222]) == (4, 3)
assert total_even_odd_count([13579, 24680]) == (5, 5)
assert total_even_odd_count([-24680, 13579]) == (5, 5)