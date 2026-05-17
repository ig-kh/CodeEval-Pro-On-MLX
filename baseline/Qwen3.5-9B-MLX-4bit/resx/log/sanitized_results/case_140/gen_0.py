
def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores, 
    and if a string has more than 2 consecutive spaces, 
    then replace all consecutive spaces with - 
    
    fix_spaces("Example") == "Example"
    fix_spaces("Example 1") == "Example_1"
    fix_spaces(" Example 2") == "_Example_2"
    fix_spaces(" Example   3") == "_Example-3"
    """
    result = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] == ' ':
            # Count consecutive spaces
            count = 0
            j = i
            while j < n and text[j] == ' ':
                count += 1
                j += 1

            if count > 2:
                result.append('-')
            else:
                result.append('_')

            i = j
        else:
            result.append(text[i])
            i += 1

    return ''.join(result)

def process_and_concatenate_strings(string_list):
    """
    Given a list of strings, replace all spaces in each string with underscores, and if a string has more than 2 consecutive spaces, then replace all consecutive spaces with a hyphen. After processing all strings, concatenate them into a fix_spaces function. Finally, replace all underscores in the concatenated string with asterisks.
    """
    processed_strings = [fix_spaces(s) for s in string_list]
    concatenated_string = '\n'.join(processed_strings)
    return concatenated_string.replace('_', '*')


assert process_and_concatenate_strings(['Example', 'Example 1', ' Example 2', ' Example   3']) == 'Example\nExample*1\n*Example*2\n*Example-3'
assert process_and_concatenate_strings(['Hello World', '  Test   Case ', 'Another Example']) == 'Hello*World\n**Test-Case*\nAnother*Example'
assert process_and_concatenate_strings(['Single']) == 'Single'
assert process_and_concatenate_strings(['   Leading spaces', 'Trailing spaces   ', ' Both   sides   ']) == '-Leading*spaces\nTrailing*spaces-\n*Both-sides-'