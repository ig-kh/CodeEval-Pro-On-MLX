
def triangle_area(a, b, c):
    '''
    Given the lengths of the three sides of a triangle. Return the area of
    the triangle rounded to 2 decimal points if the three sides form a valid triangle. 
    Otherwise return -1
    Three sides make a valid triangle when the sum of any two sides is greater 
    than the third side.
    Example:
    triangle_area(3, 4, 5) == 6.00
    triangle_area(1, 2, 10) == -1
    '''
    # Check if sides form a valid triangle
    if a + b <= c or a + c <= b or b + c <= a:
        return -1

    # Calculate semi-perimeter
    s = (a + b + c) / 2

    # Calculate area using Heron's formula
    area = (s * (s - a) * (s - b) * (s - c)) ** 0.5

    # Round to 2 decimal points
    return round(area, 2)

def total_triangle_area(triangles):
    '''
    Given a list of tuples, where each tuple contains the (a, b, c) lengths of the three sides of a triangle, 
    return the total area of all valid triangles. If no valid triangles are found, return -1. 
    Each tuple should be processed using the `triangle_area` valid_triangle_area function to determine if it forms a valid triangle and to calculate its area.
    '''
    total_area = 0
    valid_count = 0

    for a, b, c in triangles:
        area = triangle_area(a, b, c)
        if area != -1:
            total_area += area
            valid_count += 1

    # If no valid triangles are found, return -1
    if valid_count == 0:
        return -1

    return total_area


assert total_triangle_area([(3, 4, 5), (1, 2, 10)]) == 6.00
assert total_triangle_area([(5, 12, 13), (7, 24, 25)]) == 114.0
assert total_triangle_area([(1, 2, 3), (1, 1, 1)]) == 0.43
assert total_triangle_area([(1, 2, 10), (10, 20, 30)]) == -1