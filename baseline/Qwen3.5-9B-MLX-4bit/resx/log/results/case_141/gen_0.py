
def file_name_check(file_name):
    """Create a function which takes a string representing a file's name, and returns
    'Yes' if the the file's name is valid, and returns 'No' otherwise.
    A file's name is considered to be valid if and only if all the following conditions 
    are met:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the dot should not be empty, and it starts with a letter from 
    the latin alphapet ('a'-'z' and 'A'-'Z').
    - The substring after the dot should be one of these: ['txt', 'exe', 'dll']
    Examples:
    file_name_check("example.txt") # => 'Yes'
    file_name_check("1example.dll") # => 'No' (the name should start with a latin alphapet letter)
    """
    if '.' not in file_name:
        return 'No'
    
    parts = file_name.split('.')
    if len(parts) != 2:
        return 'No'
    
    name, ext = parts
    
    if not name or not name[0].isalpha():
        return 'No'
    
    if ext not in ['txt', 'exe', 'dll']:
        return 'No'
    
    digit_count = sum(1 for char in file_name if char.isdigit())
    if digit_count > 3:
        return 'No'
    
    return 'Yes'

def validate_file_names(file_names):
    """Create a function which takes a list of file names and returns a list of valid file names based on the following rules:
    - There should not be more than three digits ('0'-'9') in the file's name.
    - The file's name contains exactly one dot '.'
    - The substring before the


assert validate_file_names(['example.txt', '1example.dll', 'test.exe', 'file.doc']) == ['example.txt', 'test.exe']
assert validate_file_names(['1234.txt', 'example..txt', 'file123.dll']) == ['file123.dll']
assert validate_file_names(['.txt', 'exe.exe', 'dll.dll']) == ['exe.exe', 'dll.dll']
assert validate_file_names([]) == None
assert validate_file_names(['a1b2c3.txt', 'valid.exe', 'another.dll']) == ['a1b2c3.txt', 'valid.exe', 'another.dll']