


def sum_squares(lst):
    """"
    This function will take a list of integers. For all entries in the list, the function shall square the integer entry if its index is a 
    multiple of 3 and will cube the integer entry if its index is a multiple of 4 and not a multiple of 3. The function will not 
    change the entries in the list whose indexes are not a multiple of 3 or 4. The function shall then return the sum of all entries. 
    
    Examples:
    For lst = [1,2,3] the output should be 6
    For lst = []  the output should be 0
    For lst = [-1,-5,2,-1,-5]  the output should be -126
    """
    import re

    result = 0
    for i, num in enumerate(lst):
        if i % 3 == 0:
            result += num ** 2
        elif i % 4 == 0:
            result += num ** 3
        else:
            result += num

    return result

def total_score(games):
    """
    Given a list of lists of integers, where each sublist represents a set of scores for a game. For each game, calculate the total score using the following rules: if the game index is even, use the sum_squares function to calculate the score; if the game index is odd, use the sum_squares function twice, summing the results of both calls. Return the total score for all games.
    """
    total_score = 0

    for i, game in enumerate(games):
        if i % 2 == 0:
            total_score += sum_squares(game)
        else:
            total_score += sum_squares(sum_squares(game))

    return total_score


assert total_score([[1,2,3], [4,5,6]]) == 60
assert total_score([[1,2,3], [4,5,6], [7,8,9]]) == 126
assert total_score([[], [1,1,1]]) == 6
assert total_score([[-1,-5,2], [-1,-5]]) == -10