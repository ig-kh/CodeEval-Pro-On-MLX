
def select_words(s, n):
    """Given a string s and a natural number n, you have been tasked to implement 
    a function that returns a list of all words from string s that contain exactly 
    n consonants, in order these words appear in the string s.
    If the string s is empty then the function should return an empty list.
    Note: you may assume the input string contains only letters and spaces.
    Examples:
    select_words("Mary had a little lamb", 4) ==> ["little"]
    select_words("Mary had a little lamb", 3) ==> ["Mary", "lamb"]
    select_words("simple white space", 2) ==> []
    select_words("Hello world", 4) ==> ["world"]
    select_words("Uncle sam", 3) ==> ["Uncle"]
    """
    import re

    vowels = set("aeiouAEIOU")
    result = []

    # Split the string into words
    words = s.split()

    for word in words:
        # Count the number of consonants in the current word
        consonant_count = sum(1 for char in word if char.isalpha() and char not in vowels)

        # If the word has exactly n consonants, add it to the result list
        if consonant_count == n:
            result.append(word)

    return result

def select_words_from_list(strings, n):
    """Given a list of strings and a natural number n, you have been tasked to implement a function that returns a list of all words from each string in the list that contain exactly n consonants, in the order these words appear in their respective strings. If a string in the list is empty, it should be ignored. The function should return a list of lists, where each sublist corresponds to the words from the respective string in the input list that meet the criteria. Note: you may assume the input strings contain only letters and spaces."""
    result = []

    for string in strings:
        # Skip empty strings
        if not string:
            continue

        # Use the select_words function to get the words with exactly n consonants
        words = select_words(string, n)
        result.append(words)

    return result


assert select_words_from_list(["Mary had a little lamb", "simple white space", "Hello world", "Uncle sam"], 4) == [['little'], ['simple'], ['world'], []]
assert select_words_from_list(["Mary had a little lamb", "simple white space", "Hello world", "Uncle sam"], 3) == [['Mary', 'lamb'], ['white', 'space'], ['Hello'], ['Uncle']]
assert select_words_from_list(["", "Mary had a little lamb", "", "Hello world"], 4) == [['little'], ['world']]