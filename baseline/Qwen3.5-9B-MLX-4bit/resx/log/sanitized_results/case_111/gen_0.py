
def histogram(test):
    """Given a string representing a space separated lowercase letters, return a dictionary
    of the letter with the most repetition and containing the corresponding count.
    If several letters have the same occurrence, return all of them.
    
    Example:
    histogram('a b c') == {'a': 1, 'b': 1, 'c': 1}
    histogram('a b b a') == {'a': 2, 'b': 2}
    histogram('a b c a b') == {'a': 2, 'b': 2}
    histogram('b b b b a') == {'b': 4}
    histogram('') == {}

    """
    if not test:
        return {}

    # Split the string by spaces and count the occurrences of each letter
    letters = test.split()
    counts = {}
    for letter in letters:
        counts[letter] = counts.get(letter, 0) + 1

    # Find the maximum count
    max_count = max(counts.values()) if counts else 0

    # Return a dictionary of all letters with the maximum count
    result = {letter: count for letter, count in counts.items() if count == max_count}

    return result

def total_histogram(strings):
    """Given a list of strings representing space separated lowercase letters, return a dictionary of the letter(s) with the most total repetitions across all strings and containing the corresponding total count. If several letters have the same total occurrence, return all of them."""
    # Combine all strings into one string
    combined_string = " ".join(strings)

    # Use the histogram function to get the result for the combined string
    return histogram(combined_string)


assert total_histogram(['a b c', 'a b b a']) == {'a': 3, 'b': 3}
assert total_histogram(['b b b b a', 'a b c a b']) == {'b': 6}
assert total_histogram(['', 'a b c a b']) == {'a': 2, 'b': 2}
assert total_histogram(['a b c', 'd e f']) == {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1}
assert total_histogram([]) == {}