
FIX = """
Add more test cases.
"""

def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word.

    Example:
    >>> vowels_count("abcde")
    2
    >>> vowels_count("ACEDY")
    3
    """
    vowels = "aeiouAEIOU"
    count = 0
    for char in s:
        if char in vowels:
            count += 1
    if s and s[-1] == "y" or s[-1] == "Y":
        count += 1
    return count

def total_vowels_count(strings):
    """total_vowels_count is a function that takes a list of strings and returns the total number of vowels in all the strings combined.
    >>> total_vowels_count(["abcde", "ACEDY"])
    5
    >>> total_vowels_count(["hello", "world"])
    2
    >>> total_vowels_count([])
    0
    >>> total_vowels_count(["y", "Y"])
    2
    """
    return sum(vowels_count(s) for s in strings)


assert total_vowels_count(['abcde', 'ACEDY']) == 5
assert total_vowels_count(['hello', 'world']) == 3
assert total_vowels_count(['python', 'programming']) == 4
assert total_vowels_count(['sky', 'fly']) == 2
assert total_vowels_count(['rhythm', 'syzygy']) == 1