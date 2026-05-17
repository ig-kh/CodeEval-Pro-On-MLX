
def iscube(a):
    '''
    Write a function that takes an integer a and returns True 
    if this ingeger is a cube of some integer number.
    Note: you may assume the input is always valid.
    Examples:
    iscube(1) ==> True
    iscube(2) ==> False
    iscube(-1) ==> True
    iscube(64) ==> True
    iscube(0) ==> True
    iscube(180) ==> False
    '''
    # Check if the number is a perfect cube
    root = round(a ** (1/3))
    return root ** 3 == a

def sum_of_cubes(lst):
    '''
    Given a list of integers, write a function that returns the sum of all the integers in the list that are cubes of some integer. You may assume the input list is always valid.
    '''
    total = 0
    for num in lst:
        if iscube(num):
            total += num
    return total


assert sum_of_cubes([1, 2, 64, 180]) == 65
assert sum_of_cubes([-1, -8, -27, 0]) == -36
assert sum_of_cubes([125, 216, 343, 512]) == 1196
assert sum_of_cubes([100, 200, 300, 400]) == 0
assert sum_of_cubes([1, 8, 27, 64, 125]) == 225