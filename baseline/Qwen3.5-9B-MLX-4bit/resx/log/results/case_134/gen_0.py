
def check_if_last_char_is_a_letter(txt):
    '''
    Create a function that returns True if the last character
    of a given string is an alphabetical character and is not
    a part of a word, and False otherwise.
    Note: "word" is a group of characters separated by space.

    Examples:
    check_if_last_char_is_a_letter("apple pie") ➞ False
    check_if_last_char_is_a_letter("apple pi e") ➞ True
    check_if_last_char_is_a_letter("apple pi e ") ➞ False
    check_if_last_char_is_a_letter("") ➞ False 
    '''
    if not txt:
        return False
    
    # Check if the last character is an alphabetical character
    if not txt[-1].isalpha():
        return False
    
    # Check if the last character is not part of a word
    # If the last character is a letter, it should be preceded by a space or be the first character
    if len(txt) == 1:
        return True
    
    return txt[-2] == ' '

def check_if_last_char_is_a_letter_in_list(lst):
    '''
    Create a function that takes a list of strings and returns a list of booleans indicating whether each string in the list ends with an alphabetical character that is not part of a word. The function should handle multiple strings and return a boolean list corresponding to each string's condition.
    '''
    result = []
    for string in lst:
        result.append(check_if_last_char_is_a_letter(string))
    
    return result


assert check_if_last_char_is_a_letter_in_list(['apple pie', 'apple pi e', 'apple pi e ', '']) == [False, True, False, False]
assert check_if_last_char_is_a_letter_in_list(['hello world', 'hello world ', 'hello world!']) == [False, False, False]
assert check_if_last_char_is_a_letter_in_list(['a b c', 'a b c ', 'a b c d']) == [True, False, True]