
def right_angle_triangle(a, b, c):
    '''
    Given the lengths of the three sides of a triangle. Return True if the three
    sides form a right-angled triangle, False otherwise.
    A right-angled triangle is a triangle in which one angle is right angle or 
    90 degree.
    Example:
    right_angle_triangle(3, 4, 5) == True
    right_angle_triangle(1, 2, 3) == False
    '''
    sides = sorted([a, b, c])
    return sides[0]**2 + sides[1]**2 == sides[2]**2

def count_right_angle_triangles(triangles):
    """Given a list of tuples, where each tuple contains three integers representing the sides of a triangle, determine the number of right-angled triangles in the

assert count_right_angle_triangles([(3, 4, 5), (1, 2, 3), (5, 12, 13)]) == 2
assert count_right_angle_triangles([(1, 1, 1), (2, 2, 2), (3, 3, 3)]) == 0
assert count_right_angle_triangles([(6, 8, 10), (7, 24, 25), (8, 15, 17)]) == 3
assert count_right_angle_triangles([(10, 20, 30), (15, 20, 25), (20, 21, 29)]) == 2