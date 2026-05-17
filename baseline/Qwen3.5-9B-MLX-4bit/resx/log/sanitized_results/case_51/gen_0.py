

def remove_vowels(text):
    """
    remove_vowels is a function that takes string and returns string without vowels.
    >>> remove_vowels('')
    ''
    >>> remove_vowels("abcdef\nghijklm")
    'bcdf\nghjklm'
    >>> remove_vowels('abcdef')
    'bcdf'
    >>> remove_vowels('aaaaa')
    ''
    >>> remove_vowels('aaBAA')
    'B'
    >>> remove_vowels('zbcd')
    'zbcd'
    """
    vowels = set("aeiouAEIOU")
    return "".join([ch for ch in text if ch not in vowels])


def remove_vowels_from_list(texts):
    """
    Create a function `remove_vowels_from_list` that takes a list of strings and returns a list of strings with all vowels removed from each string. The function should handle an empty list and lists containing empty strings.
    """
    return [remove_vowels(text) for text in texts]


assert remove_vowels_from_list(['']) == ['']
assert remove_vowels_from_list(['abcdefghijklm', 'abcdef']) == ['bcdfghjklm', 'bcdf']
assert remove_vowels_from_list(['aaaaa', 'aaBAA']) == ['', 'B']
assert remove_vowels_from_list(['zbcd', '']) == ['zbcd', '']
assert remove_vowels_from_list([]) == []