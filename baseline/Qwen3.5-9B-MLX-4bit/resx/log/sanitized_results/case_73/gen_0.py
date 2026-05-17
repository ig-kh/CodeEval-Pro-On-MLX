
def smallest_change(arr):
    """
    Given an array arr of integers, find the minimum number of elements that
    need to be changed to make the array palindromic. A palindromic array is an array that
    is read the same backwards and forwards. In one change, you can change one element to any other element.

    For example:
    smallest_change([1,2,3,5,4,7,9,6]) == 4
    smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
    smallest_change([1, 2, 3, 2, 1]) == 0
    """
    n = len(arr)
    changes = 0

    # Check pairs from the start and end moving towards the center
    for i in range(n // 2):
        if arr[i] != arr[n - 1 - i]:
            changes += 1

    return changes

def smallest_change_2d(arr):
    """
    Given a 2D array of integers, find the minimum number of elements that need to be changed to make each row and each column of the array
    palindromic. A palindromic row or column is a row or column that is read the same backwards and forwards. In one change, you can change one element to any other element.
    """
    if not arr or not all(isinstance(row, list) for row in arr):
        return 0

    changes = 0

    # Process each row
    for row in arr:
        changes += smallest_change(row)

    # Process each column
    num_cols = len(arr[0]) if arr else 0
    for col_idx in range(num_cols):
        column = [arr[row_idx][col_idx] for row_idx in range(len(arr))]
        changes += smallest_change(column)

    return changes


assert smallest_change_2d([[1,2,3,5,4,7,9,6], [1,2,3,4,3,2,2], [1,2,3,2,1]]) == 7
assert smallest_change_2d([[1,2,3,4], [4,3,2,1], [1,2,2,1], [1,2,3,4]]) == 8
assert smallest_change_2d([[1,2,3,2,1], [1,2,3,2,1], [1,2,3,2,1], [1,2,3,2,1]]) == 0