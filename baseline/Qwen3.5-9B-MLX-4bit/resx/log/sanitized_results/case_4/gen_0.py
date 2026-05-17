from typing import List


def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
    import re
    from typing import List

    if not numbers:
        return 0.0

    mean = sum(numbers) / len(numbers)

    total_deviation = 0.0
    for num in numbers:
        total_deviation += abs(num - mean)

    return total_deviation / len(numbers)


def overall_mean_absolute_deviation(list_of_lists: List[List[float]]) -> float:
    """ Given a list of lists of numbers, calculate the Mean Absolute Deviation (MAD) for each sublist and then find the overall Mean Absolute Deviation of these MADs. The overall MAD should be calculated around the mean of the MADs of the sublists.
    >>> overall_mean_absolute_deviation([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    1.0
    """
    if not list_of_lists:
        return 0.0

    mad_values = [mean_absolute_deviation(sublist) for sublist in list_of_lists]

    return mean_absolute_deviation(mad_values)


assert overall_mean_absolute_deviation([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]) == 0.0
assert overall_mean_absolute_deviation([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]) == 0.0
assert overall_mean_absolute_deviation([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]) == 0.0