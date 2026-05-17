
def fruit_distribution(s,n):
    """
    In this task, you will be given a string that represents a number of apples and oranges 
    that are distributed in a basket of fruit this basket contains 
    apples, oranges, and mango fruits. Given the string that represents the total number of 
    the oranges and apples and an integer that represent the total number of the fruits 
    in the basket return the number of the mango fruits in the basket.
    for examble:
    fruit_distribution("5 apples and 6 oranges", 19) ->19 - 5 - 6 = 8
    fruit_distribution("0 apples and 1 oranges",3) -> 3 - 0 - 1 = 2
    fruit_distribution("2 apples and 3 oranges", 100) -> 100 - 2 - 3 = 95
    fruit_distribution("100 apples and 1 oranges",120) -> 120 - 100 - 1 = 19
    """
    import re
    apples = int(re.search(r'(\d+)\s+apples', s).group(1))
    oranges = int(re.search(r'(\d+)\s+oranges', s).group(1))
    return n - apples - oranges

def total_mangoes(distributions, totals):
    """total_mangoes is a function that takes a list of fruit distribution strings and a list of total fruit counts, and returns the total number of mango fruits across all baskets.
    >>> total_mangoes(["5 apples and 6 oranges", "0 apples and 1 oranges"], [19, 3])
    10
    >>> total_mangoes(["2 apples and 3 oranges", "100 apples and 1 oranges"], [100, 120])
    97
    >>> total_mangoes(["5 apples and 6 oranges", "0 apples and 1 oranges"], [19, 3])
    10
    >>> total_mangoes([], [])
    0
    """
    return sum(fruit_distribution(d, t) for d, t in zip(distributions, totals))


assert total_mangoes(["5 apples and 6 oranges", "0 apples and 1 oranges", "2 apples and 3 oranges", "100 apples and 1 oranges"], [19, 3, 100, 120]) == 124