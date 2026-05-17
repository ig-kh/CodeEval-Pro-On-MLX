
def words_string(s):
    """
    You will be given a string of words separated by commas or spaces. Your task is
    to split the string into words and return an array of the words.
    
    For example:
    words_string("Hi, my name is John") == ["Hi", "my", "name", "is", "John"]
    words_string("One, two, three, four, five, six") == ["One", "two", "three", "four", "five", "six"]
    """
    import re
    return re.findall(r'\w+', s)

def sentences_to_words(sentences):
    """
    Given a list of sentences, each separated by a newline character, your task is to split each sentence into words and return a list of lists where each inner list contains the words of a sentence. The sentences may contain words separated by commas or spaces.
    """
    result = []
    for sentence in sentences:
        result.append(words_string(sentence))
    return result


assert sentences_to_words("Hi, my name is John\nOne, two, three, four, five, six") == [['Hi', 'my', 'name', 'is', 'John'], ['One', 'two', 'three', 'four', 'five', 'six']]
assert sentences_to_words("Hello world\nThis is a test") == [['Hello', 'world'], ['This', 'is', 'a', 'test']]