
def compare(game,guess):
    """I think we all remember that feeling when the result of some long-awaited
    event is finally known. The feelings and thoughts you have at that moment are
    definitely worth noting down and comparing.
    Your task is to determine if a person correctly guessed the results of a number of matches.
    You are given two arrays of scores and guesses of equal length, where each index shows a match. 
    Return an array of the same length denoting how far off each guess was. If they have guessed correctly,
    the value is 0, and if not, the value is the absolute difference between the guess and the score.
    
    
    example:

    compare([1,2,3,4,5,1],[1,2,3,4,2,-2]) -> [0,0,0,0,3,3]
    compare([0,5,0,0,0,4],[4,1,1,0,0,-2]) -> [4,4,1,0,0,6]
    """
    import re

    result = []
    for score, guess in zip(game, guess):
        diff = abs(score - guess)
        result.append(diff if diff != 0 else 0)
    return result

def compare_matches(matches):
    """You are given a list of matches with their actual scores and a list of guesses for each match. Each match has multiple rounds, and you need to determine how far off each guess was for each round. The input consists of a list of matches, where each match is represented by a list containing the actual scores and the guesses for each round. Your task is to return a list of lists, where each sublist corresponds to a match and contains the differences for each round. If the `compare` function.
    """
    result = []
    for match in matches:
        scores, guesses = match
        diff_list = compare(scores, guesses)
        result.append(diff_list)
    return result


assert compare_matches([([1, 2, 3], [1, 2, 3]), ([4, 5, 6], [4, 5, 6])]) == [[0, 0, 0], [0, 0, 0]]
assert compare_matches([([1, 2, 3], [1, 2, 4]), ([4, 5, 6], [4, 5, 7])]) == [[0, 0, 1], [0, 0, 1]]
assert compare_matches([([1, 2, 3], [1, 2, 4]), ([4, 5, 6], [4, 5, 7])]) == [[0, 0, 1], [0, 0, 1]]
assert compare_matches([([1, 2, 3], [1, 2, 4]), ([4, 5, 6], [4, 5, 7])]) == [[0, 0, 1], [0, 0, 1]]
assert compare_matches([([1, 2, 3], [1, 2, 4]), ([4, 5, 6], [4, 5, 7])]) == [[0, 0, 1], [0, 0, 1]]