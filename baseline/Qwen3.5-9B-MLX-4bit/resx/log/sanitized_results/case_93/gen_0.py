
def encode(message):
    """
    Write a function that takes a message, and encodes in such a 
    way that it swaps case of all letters, replaces all vowels in 
    the message with the letter that appears 2 places ahead of that 
    vowel in the english alphabet. 
    Assume only letters. 
    
    Examples:
    >>> encode('test')
    'TGST'
    >>> encode('This is a message')
    'tHKS KS C MGSSCGG'
    """
    import math

    vowels = 'aeiouAEIOU'
    result = []

    for char in message:
        if char.isalpha():
            # Swap case
            swapped_char = char.swapcase()

            # Check if the swapped character is a vowel (after swapping)
            if swapped_char in vowels:
                # Replace with the letter 2 places ahead
                index = vowels.index(swapped_char)
                new_index = (index + 2) % 26
                replaced_char = vowels[new_index]
                result.append(replaced_char)
            else:
                result.append(swapped_char)
        else:
            result.append(char)

    return ''.join(result)

def encode_and_concatenate(messages):
    """
    Given a list of messages, encode each message using the `encode` `encode_and_concatenate` function and then concatenate all the encoded messages into a single string. Finally, reverse the concatenated string and return it.
    """
    encoded_messages = [encode(msg) for msg in messages]
    concatenated = ''.join(encoded_messages)
    return concatenated[::-1]


assert encode_and_concatenate(['test', 'This is a message']) == 'GGCSSGM C SK SKHtTSGT'
assert encode_and_concatenate(['hello', 'world']) == 'DLRQWQLLGH'
assert encode_and_concatenate(['Python', 'Programming']) == 'GNKMMCRGQRpNQHTYp'
assert encode_and_concatenate(['OpenAI', 'GPT-3']) == '3-tpgkcNGPq'
assert encode_and_concatenate(['example', 'problem']) == 'MGLBQRPGLPMCXG'