
def is_nested(string):
    '''
    Create a function that takes a string as input which contains only square brackets.
    The function should return True if and only if there is a valid subsequence of brackets 
    where at least one bracket in the subsequence is nested.

    is_nested('[[]]') ➞ True
    is_nested('[]]]]]]][[[[[]') ➞ False
    is_nested('[][]') ➞ False
    is_nested('[]') ➞ False
    is_nested('[[][]]') ➞ True
    is_nested('[[]][[') ➞ True
    '''
    import re

    # Count the balance of brackets
    balance = 0
    max_balance = 0
    for char in string:
        if char == '[':
            balance += 1
        else:
            balance -= 1

        # If balance drops below zero, the string is invalid
        if balance < 0:
            return False

        max_balance = max(max_balance, balance)

    # For a valid bracket sequence, the final balance must be zero
    if balance != 0:
        return False

    # Check for nesting: a valid sequence is nested if max_balance > 1
    return max_balance > 1

def is_nested_in_list(strings):
    '''
    Create a function that takes a list of strings, each containing only square brackets. 
    The function should return True if and only if there is at least one valid subsequence of brackets 
    in the list where at least one bracket in the subsequence is nested in at least one of the strings. 
    Otherwise, it should return False.
    '''
    # Helper function to check if a string is valid and has nesting
    def is_valid_nested(s):
        return is_nested(s)

    # Check each string in the list
    for s in strings:
        if is_valid_nested(s):
            return True

    return False


assert is_nested_in_list(['[[]]', '[]', '[][]']) == True
assert is_nested_in_list(['[]', '[][]', '[]']) == False
assert is_nested_in_list(['[[]]', '[[[]]]', '[][]']) == True
assert is_nested_in_list(['[]', '[]', '[]']) == False
assert is_nested_in_list(['[[]]', '[]', '[][]']) == True