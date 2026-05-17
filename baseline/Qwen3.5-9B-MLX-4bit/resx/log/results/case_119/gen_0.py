
def match_parens(lst):
    '''
    You are given a list of two strings, both strings consist of open
    parentheses '(' or close parentheses ')' only.
    Your job is to check if it is possible to concatenate the two strings in
    some order, that the resulting string will be good.
    A string S is considered to be good if and only if all parentheses in S
    are balanced. For example: the string '(())()' is good, while the string
    '())' is not.
    Return 'Yes' if there's a way to make a good string, and return 'No' otherwise.

    Examples:
    match_parens(['()(', ')']) == 'Yes'
    match_parens([')', ')']) == 'No'
    '''
    def is_balanced(s):
        count = 0
        for char in s:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    # Check if the original order is balanced
    combined = lst[0] + lst[1]
    if is_balanced(combined):
        return 'Yes'
    
    # Check if the reversed order is balanced
    combined = lst[1] + lst[0]
    if is_balanced(combined):
        return 'Yes'
    
    return 'No'

def can_form_good_string(lst):
    """You are given a list of strings, each string consists of open parentheses '(' or close parentheses ')' only. Your job is to determine if it is possible to concatenate any number of these strings in some order, such that the resulting string will be good. A string S is considered to be good if and only if all parentheses in S are balanced. For example, the string '(())()' is good, while the string '())' is not. Return 'Yes' if there's a way to make a good string by concatenating any

assert can_form_good_string(['()(', ')']) == 'Yes'
assert can_form_good_string([')', ')']) == 'No'
assert can_form_good_string(['(', '())']) == 'Yes'
assert can_form_good_string(['()', '()']) == 'Yes'
assert can_form_good_string(['((', '))']) == 'Yes'