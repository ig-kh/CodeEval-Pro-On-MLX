
def words_in_sentence(sentence):
    """
    You are given a string representing a sentence,
    the sentence contains some words separated by a space,
    and you have to return a string that contains the words from the original sentence,
    whose lengths are prime numbers,
    the order of the words in the new string should be the same as the original one.

    Example 1:
        Input: sentence = "This is a test"
        Output: "is"

    Example 2:
        Input: sentence = "lets go for swimming"
        Output: "go for"

    Constraints:
        * 1 <= len(sentence) <= 100
        * sentence contains only letters
    """
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    words = sentence.split()
    filtered_words = [word for word in words if is_prime(len(word))]
    return ' '.join(filtered_words)

def words_in_sentences(sentences):
    """
    Given a list of sentences, return a list of strings where each string contains words from the corresponding sentence whose lengths are prime numbers. The order of the words in each string should be the same as in the original sentence.
    """
    result = []
    for sentence in sentences:
        result.append(words_in_sentence(sentence))
    
    return result


assert words_in_sentences([]) == []  # Test with an empty list
assert words_in_sentences([""]) == ['']  # Test with an empty string
assert words_in_sentences(["This is a test", "lets go for swimming"]) == ["is", "go for"]
assert words_in_sentences(["hello world", "python programming is fun"]) == ['hello world', 'programming is fun']
assert words_in_sentences(["a", "ab", "abc", "abcd"]) == ['', 'ab', 'abc', '']
assert words_in_sentences(["prime numbers are fascinating", "two three five seven eleven"]) == ['prime numbers are fascinating', 'two three seven']