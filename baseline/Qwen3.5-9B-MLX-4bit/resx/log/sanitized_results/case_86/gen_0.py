
def anti_shuffle(s):
    """
    Write a function that takes a string and returns an ordered version of it.
    Ordered version of string, is a string where all words (separated by space)
    are replaced by a new word where all the characters arranged in
    ascending order based on ascii value.
    Note: You should keep the order of words and blank spaces in the sentence.

    For example:
    anti_shuffle('Hi') returns 'Hi'
    anti_shuffle('hello') returns 'ehllo'
    anti_shuffle('Hello World!!!') returns 'Hello !!!Wdlor'
    """
    import random

    words = s.split(' ')
    result = []

    for word in words:
        sorted_word = ''.join(sorted(word))
        result.append(sorted_word)

    return ' '.join(result)

def multi_anti_shuffle(string_list):
    """Write a function that takes a list of strings and returns a new list where each string is the anti-shuffled version of the corresponding string in the input list. Additionally,

assert multi_anti_shuffle(['Hi', 'hello', 'Hello World!!!']) == (['Hi', 'ehllo', 'Hello !!!Wdlor'], 'Hi ehllo Hello !!!Wdlor')
assert multi_anti_shuffle(['', 'a', 'ab', 'abc']) == (['', 'a', 'ab', 'abc'], ' a ab abc')
assert multi_anti_shuffle([]) == ([], '')