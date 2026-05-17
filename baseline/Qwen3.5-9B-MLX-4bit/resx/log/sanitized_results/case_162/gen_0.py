
def string_to_md5(text):
    """
    Given a string 'text', return its md5 hash equivalent string.
    If 'text' is an empty string, return None.

    >>> string_to_md5('Hello world') == '3e25960a79dbc69b674cd4ec67a72c62'
    """
    import hashlib

    if not text:
        return None

    return hashlib.md5(text.encode()).hexdigest()

def list_to_md5_dict(string_list):
    """
    Given a list of strings, return a dictionary where each key is a string from the list and the value is the md5 hash of the string. If a string is empty, its corresponding value is None. Additionally, if the list contains duplicate strings, the dictionary should only include the first occurrence of each string.
    """
    import hashlib

    md5_dict = {}
    seen_strings = set()

    for string in string_list:
        if string in seen_strings:
            continue

        if not string:
            md5_dict[string] = None
        else:
            md5_dict[string] = string_to_md5(string)

        seen_strings.add(string)

    return md5_dict


assert list_to_md5_dict(['Hello world', '', 'Python', 'Hello world']) == {'Hello world': '3e25960a79dbc69b674cd4ec67a72c62', '': None, 'Python': 'a7f5f35426b927411fc9231b56382173'}