

def triangle_area(a, h):
    """Given length of a side and high return area for a triangle.
    >>> triangle_area(5, 3)
    7.5
    """
    return 0.5 * a * h

def total_triangle_area(triangles):
    """
    Given a list of tuples where each tuple contains the length of a side and the corresponding height 
    of multiple triangles, calculate the total area of all the triangles combined.

    >>> total_triangle_area([(5, 3), (4, 6), (10, 2)])
    25.0
    """
    total_area = 0.0
    for side, height in triangles:
        total_area += triangle_area(side, height)
    return total_area


assert total_triangle_area([(5, 3), (4, 2), (6, 4)]) == 23.5
assert total_triangle_area([(1, 1), (2, 2), (3, 3)]) == 7.0
assert total_triangle_area([]) == 0.0