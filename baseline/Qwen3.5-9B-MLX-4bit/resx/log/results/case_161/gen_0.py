
def solve(s):
    """You are given a string s.
    if s[i] is a letter, reverse its case from lower to upper or vise versa, 
    otherwise keep it as it is.
    If the string contains no letters, reverse the string.
    The function should return the resulted string.
    Examples
    solve("1234") = "4321"
    solve("ab") = "AB"
    solve("#a@C") = "#A@c"
    """
    has_letter = any(c.isalpha() for c in s)
    
    if has_letter:
        return ''.join(c.upper() if c.islower() else c.lower() for c in s)
    else:
        return s[::-1]

def new_solve(strings):
    """
    Given a list of strings, process each string using the solve function.
    If a string contains at least one letter, reverse its case for all letters in the string.
    If a string contains no letters, reverse the entire string.
    After processing all strings, concatenate the results into a multiple calls to the first solution.
    Finally, reverse the case of all letters in the concatenated result.
    Return the final string.
    """
    # Step 1: Process each string using solve function
    processed_strings = [solve(s) for s in strings]
    
    # Step 2: Concatenate the processed strings
    concatenated = ''.join(processed_strings)
    
    # Step 3: Reverse the case of all letters in the concatenated result
    final_result = ''.join(c.upper() if c.islower() 
                          else c.lower() if c.isalpha() else c for c in concatenated)
    
    return final_result


assert new_solve(['1234', 'ab', '#a@C']) == '4321ab#a@C'
assert new_solve(['#', 'a', 'B']) == '#aB'
assert new_solve(['', '1', 'A']) == '1A'
assert new_solve(['abc', 'DEF', '123']) == 'abcDEF321'