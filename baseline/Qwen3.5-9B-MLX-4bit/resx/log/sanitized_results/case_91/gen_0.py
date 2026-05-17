
def is_bored(S):
    """
    You'll be given a string of words, and your task is to count the number
    of boredoms. A boredom is a sentence that starts with the word "I".
    Sentences are delimited by '.', '?' or '!'.
   
    For example:
    >>> is_bored("Hello world")
    0
    >>> is_bored("The sky is blue. The sun is shining. I love this weather")
    1
    """
    count = 0
    sentences = S.split('.')
    for sentence in sentences:
        if sentence.strip().startswith('I'):
            count += 1

    return count

def count_total_boredoms(paragraphs):
    """
    Given a list of strings, each representing a paragraph, count the total number of boredoms across all paragraphs. A boredom is defined as a sentence that starts with the


assert count_total_boredoms(['The sky is blue. The sun is shining. I love this weather', 'I am happy. I am excited.']) == 3
assert count_total_boredoms(['Hello world', 'This is a test.']) == 0
assert count_total_boredoms(['I am bored. I need a break.', 'The weather is nice.']) == 2
assert count_total_boredoms(['I am excited. I am ready.', 'I am tired.']) == 3
assert count_total_boredoms(['This is a test. I am testing.']) == 1