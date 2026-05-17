

def is_palindrome(text: str):
    """
    Checks if given string is a palindrome
    >>> is_palindrome('')
    True
    >>> is_palindrome('aba')
    True
    >>> is_palindrome('aaaaa')
    True
    >>> is_palindrome('zbcd')
    False
    """
    return text == text[::-1]

def count_palindromic_substrings(strings: list) -> list:
    """
    Given a list of strings, return the number of palindromic substrings in each string. 
    A palindromic substring is a contiguous substring that reads the same forward and backward. 
    For each string, find all possible substrings and count how many of them are palindromes.
    
    >>> count_palindromic_substrings(['', 'aba', 'aaaaa', 'zbcd'])
    [1, 4, 6, 0]
    """
    results = []
    for s in strings:
        count = 0
        # Iterate over all possible substrings
        for i in range(len(s)):
            for j in range(i + 1, len(s) + 1):
                substring = s[i:j]
                if is_palindrome(substring):
                    count += 1
        results.append(count)
    return results


assert count_palindromic_substrings(['a', 'ab', 'aa']) == [1, 2, 3]
assert count_palindromic_substrings(['abc', 'aaa', '']) == [3, 6, 0]
assert count_palindromic_substrings(['aba', 'abba', 'abcba']) == [4, 6, 7]