

def median(l: list):
    """Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    """
    if not l:
        raise ValueError("The list must not be empty.")
    
    sorted_l = sorted(l)
    n = len(sorted_l)
    
    if n % 2 == 1:
        return sorted_l[n // 2]
    else:
        return (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0

def median_of_medians(list_of_lists: list):
    """
    Given a list of lists, where each sublist contains numbers, write a function to return the median 
    of the medians of each sublist.
    
    >>> median_of_medians([[1, 3, 2], [4, 5, 6]])
    4.0
    >>> median_of_medians([[10, 20], [5, 15, 25]])
    17.5
    """
    if not list_of_lists:
        raise ValueError("The list of lists must not be empty.")
    
    medians = []
    for sublist in list_of_lists:
        if not sublist:
            raise ValueError("Each sublist must contain at least one element.")
        medians.append(median(sublist))
    
    return median(medians)


assert median_of_medians([[3, 1, 2, 4, 5], [-10, 4, 6, 1000, 10, 20], [7, 8, 9]]) == 8.0
assert median_of_medians([[1], [2], [3], [4], [5]]) == 3
assert median_of_medians([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == 5
assert median_of_medians([[10, 20, 30], [40, 50, 60], [70, 80, 90]]) == 50.0
assert median_of_medians([[1, 3, 5], [2, 4, 6], [7, 9, 11]]) == 4