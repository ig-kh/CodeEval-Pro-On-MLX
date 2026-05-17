
def count_upper(s):
    """
    Given a string s, count the number of uppercase vowels in even indices.
    
    For example:
    count_upper('aBCdEf') returns 1
    count_upper('abcdefg') returns 0
    count_upper('dBBE') returns 0
    """
    uppercase_vowels = "AEIOU"
    count = 0

    for i, char in enumerate(s):
        if i % 2 == 0 and char in uppercase_vowels:
            count += 1

    return count

def total_count_upper(strings):
    """
    Given a list of strings, count the total number of uppercase vowels in even indices across all strings. If a string is empty, it should be ignored in the count.
    """
    total_count = 0

    for string in strings:
        if len(string) > 0:
            total_count += count_upper(string)

    return total_count


assert total_count_upper(['aBCdEf', 'abcdefg', 'dBBE']) == 1
assert total_count_upper(['ABC', 'DEF', '']) == 1
assert total_count_upper(['', 'aEiOu', 'BcD']) == 0
assert total_count_upper(['a', 'b', 'c']) == 0
assert total_count_upper(['AeIoU', 'BcD', 'EfGhI']) == 5