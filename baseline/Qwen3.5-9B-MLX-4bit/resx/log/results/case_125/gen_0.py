
def split_words(txt):
    '''
    Given a string of words, return a list of words split on whitespace, if no whitespaces exists in the text you
    should split on commas ',' if no commas exists you should return the number of lower-case letters with odd order in the
    alphabet, ord('a') = 0, ord('b') = 1, ... ord('z') = 25
    Examples
    split_words("Hello world!") ➞ ["Hello", "world!"]
    split_words("Hello,world!") ➞ ["Hello", "world!"]
    split_words("abcdef") == 3 
    '''
    # Check if there are any whitespaces
    if ' ' in txt:
        return txt.split()
    
    # Check if there are any commas
    if ',' in txt:
        return txt.split(',')
    
    # Count lower-case letters with odd order in the alphabet
    count = 0
    for char in txt:
        if 'a' <= char <= 'z':
            # Calculate the order of the letter (0 for 'a', 1 for 'b', etc.)
            order = ord(char) - ord('a')
            if order % 2 == 1:
                count += 1
    
    return count

def process_strings(string_list):
    """
    Given a list of strings, return a dictionary where each key is a string from the list and the value is a list of words split on whitespace, if no whitespaces exist in the text, split on commas ','. If no commas exist, the value should be the number of lower-case letters with odd order in the alphabet. Additionally, the dictionary should include a special key 'total_odd_letters' which is the sum of all odd order letters from all strings.
    """
    result = {}
    total_odd_letters = 0
    
    for string in string_list:
        # Call the split_words function to get the result for each string
        processed = split_words(string)
        
        # If the result is a list, add it to the dictionary
        if isinstance(processed, list):
            result[string] = processed
        else:
            # If the result is an integer, add it to the dictionary and update total_odd_letters
            result[string] = processed
            total_odd_letters += processed
    
    result['total_odd_letters'] = total_odd_letters
    
    return result


assert process_strings(['Hello world!', 'Hello,world!', 'abcdef']) == {'Hello world!': ['Hello', 'world!'], 'Hello,world!': ['Hello', 'world!'], 'abcdef': 3, 'total_odd_letters': 3}
assert process_strings(['abc def', 'ghi,jkl', 'mno']) == {'abc def': ['abc', 'def'], 'ghi,jkl': ['ghi', 'jkl'], 'mno': 1, 'total_odd_letters': 1}
assert process_strings(['pqr', 'stu,vwx', 'yz']) == {'pqr': 2, 'stu,vwx': ['stu', 'vwx'], 'yz': 1, 'total_odd_letters': 3}