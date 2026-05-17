
def digitSum(s):
    """Task
    Write a function that takes a string as input and returns the sum of the upper characters only'
    ASCII codes.

    Examples:
        digitSum("") => 0
        digitSum("abAB") => 131
        digitSum("abcCd") => 67
        digitSum("helloE") => 69
        digitSum("woArBld") => 131
        digitSum("aAaaaXa") => 153
    """
    total = 0
    for char in s:
        if 'A' <= char <= 'Z':
            total += ord(char)
    return total

def totalUppercaseSum(strings):
    """totalUppercaseSum is a function that takes a list of strings and returns the total sum of the ASCII codes of the uppercase characters in all the strings.
    >>> totalUppercaseSum(["abAB", "abcCd", ""])
    198
    >>> totalUppercaseSum(["helloE", "woArBld"])
    200
    >>> totalUppercaseSum(["aAaaaXa"])
    153
    >>> totalUppercaseSum([])
    0
    """
    return sum(digitSum(s) for s in strings)


assert totalUppercaseSum(['abAB', 'abcCd', 'helloE']) == 267
assert totalUppercaseSum(['woArBld', 'aAaaaXa', '']) == 284
assert totalUppercaseSum(['', '']) == 0
assert totalUppercaseSum(['ABC', 'DEF']) == 405
assert totalUppercaseSum(['a', 'b', 'c']) == 0