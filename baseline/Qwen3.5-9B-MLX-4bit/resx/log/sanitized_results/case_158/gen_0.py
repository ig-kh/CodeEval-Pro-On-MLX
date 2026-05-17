
def find_max(words):
    """Write a function that accepts a list of strings.
    The list contains different words. Return the word with maximum number
    of unique characters. If multiple strings have maximum number of unique
    characters, return the one which comes first in lexicographical order.

    find_max(["name", "of", "string"]) == "string"
    find_max(["name", "enam", "game"]) == "enam"
    find_max(["aaaaaaa", "bb" ,"cc"]) == ""aaaaaaa"
    """
    import re

    def count_unique_chars(word):
        return len(set(word))

    max_unique = -1
    result = ""

    for word in words:
        unique_count = count_unique_chars(word)
        if unique_count > max_unique or (unique_count == max_unique and word < result):
            max_unique = unique_count
            result = word

    return result

def find_max_in_categories(categories):
    """Given a list of lists of strings, where each sublist represents a category of words, write a function that returns a list of the words with the maximum number of unique characters from each category. If multiple words in a category have the maximum number of unique characters, return the one which comes first in lexicographical order. If a category is empty, append 'None' to the result list.
    """
    result = []
    for category in categories:
        if not category:
            result.append('None')
        else:
            result.append(find_max(category))
    return result


assert find_max_in_categories([['name', 'of', 'string'], ['name', 'enam', 'game'], ['aaaaaaa', 'bb', 'cc']]) == ['string', 'enam', 'aaaaaaa']
assert find_max_in_categories([[], ['apple', 'banana', 'cherry'], ['dog', 'elephant']]) == ['None', 'cherry', 'elephant']
assert find_max_in_categories([['a', 'b', 'c'], [], ['x', 'y', 'z']]) == ['a', 'None', 'x']