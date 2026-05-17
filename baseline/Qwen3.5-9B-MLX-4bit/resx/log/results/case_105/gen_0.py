
def by_length(arr):
    """
    Given an array of integers, sort the integers that are between 1 and 9 inclusive,
    reverse the resulting array, and then replace each digit by its corresponding name from
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".

    For example:
      arr = [2, 1, 1, 4, 5, 8, 2, 3]   
            -> sort arr -> [1, 1, 2, 2, 3, 4, 5, 8] 
            -> reverse arr -> [8, 5, 4, 3, 2, 2, 1, 1]
      return ["Eight", "Five", "Four", "Three", "Two", "Two", "One", "One"]
    
      If the array is empty, return an empty array:
      arr = []
      return []
    
      If the array has any strange number ignore it:
      arr = [1, -1 , 55] 
            -> sort arr -> [-1, 1, 55]
            -> reverse arr -> [55, 1, -1]
      return = ['One']
    """
    digit_to_name = {
        1: "One", 2: "Two", 3: "Three", 4: "Four",
        5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
    }
    
    result = []
    for num in arr:
        if 1 <= num <= 9:
            result.append(digit_to_name[num])
    
    result.sort(key=lambda x: digit_to_name.index(x) if x in digit_to_name else float('inf'))
    result.reverse()
    
    return result

def process_arrays(arrays):
    """
    Given a list of arrays of integers, each containing integers between 1 and 9 inclusive, sort each array, reverse the resulting array, and then replace each digit by its corresponding name from 'One', '

assert process_arrays([[2, 1, 1, 4, 5, 8, 2, 3], [9, 7, 6], [1, -1, 55]]) == ['Eight', 'Five', 'Four', 'Three', 'Two', 'Two', 'One', 'One', 'Nine', 'Seven', 'Six', 'One']
assert process_arrays([[], [1, 2, 3], [4, 5, 6]]) == ['Three', 'Two', 'One', 'Six', 'Five', 'Four']
assert process_arrays([[9, 9, 9], [1, 1, 1]]) == ['Nine', 'Nine', 'Nine', 'One', 'One', 'One']
assert process_arrays([[7, 7, 7], []]) == ['Seven', 'Seven', 'Seven']
assert process_arrays([[8, 8, 8], [2, 2, 2]]) == ['Eight', 'Eight', 'Eight', 'Two', 'Two', 'Two']