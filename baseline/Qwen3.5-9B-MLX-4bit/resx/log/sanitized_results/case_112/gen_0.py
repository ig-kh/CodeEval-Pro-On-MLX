
def reverse_delete(s,c):
    """Task
    We are given two strings s and c, you have to deleted all the characters in s that are equal to any character in c
    then check if the result string is palindrome.
    A string is called palindrome if it reads the same backward as forward.
    You should return a tuple containing the result string and True/False for the check.
    Example
    For s = "abcde", c = "ae", the result should be ('bcd',False)
    For s = "abcdef", c = "b"  the result should be ('acdef',False)
    For s = "abcdedcba", c = "ab", the result should be ('cdedc',True)
    """
    # Create a set of characters to delete
    chars_to_delete = set(c)

    # Delete all characters in s that are equal to any character in c
    result = ''.join(char for char in s if char not in chars_to_delete)

    # Check if the result string is a palindrome
    is_pal = result == result[::-1]

    return (result, is_pal)

def check_palindromes_with_delete(strings):
    """Given a list of tuples where each tuple contains two strings, s and c, perform the following operations for each tuple:
    1. Delete all characters in s that are equal to any character in c.
    2. Check if the resulting string is a palindrome.
    Return a list of tuples containing the result string and whether it is a palindrome.
    """
    results = []
    for s, c in strings:
        result_string, is_palindrome = reverse_delete(s, c)
        results.append((result_string, is_palindrome))
    return results


assert process_strings([('abcde', 'ae'), ('abcdef', 'b'), ('abcdedcba', 'ab')]) == 1
assert process_strings([('racecar', ''), ('hello', 'ho'), ('level', 'v')]) == 2
assert process_strings([('programming', 'g'), ('madam', 'd'), ('python', 'y')]) == 1