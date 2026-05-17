
def will_it_fly(q,w):
    '''
    Write a function that returns True if the object q will fly, and False otherwise.
    The object q will fly if it's balanced (it is a palindromic list) and the sum of its elements is less than or equal the maximum possible weight w.

    Example:
    will_it_fly([1, 2], 5) ➞ False 
    # 1+2 is less than the maximum possible weight, but it's unbalanced.

    will_it_fly([3, 2, 3], 1) ➞ False
    # it's balanced, but 3+2+3 is more than the maximum possible weight.

    will_it_fly([3, 2, 3], 9) ➞ True
    # 3+2+3 is less than the maximum possible weight, and it's balanced.

    will_it_fly([3], 5) ➞ True
    # 3 is less than the maximum possible weight, and it's balanced.
    '''
    # Check if the object is balanced (palindromic list)
    if q != q[::-1]:
        return False
    
    # Check if the sum of elements is less than or equal to the maximum possible weight
    if sum(q) <= w:
        return True
    
    return False

def can_objects_fly(objects, weights):
    '''
    Given a list of objects and their maximum possible weights, determine if each object can fly according to the rules of the `will_it_fly` function. Return a list of boolean values indicating whether each object can fly or not. If the input list is empty, return an empty list.
    '''
    if len(objects) != len(weights):
        raise ValueError("The number of objects and weights must be equal.")
    
    results = []
    for obj, weight in zip(objects, weights):
        results.append(will_it_fly(obj, weight))
    
    return results


assert can_objects_fly([[1, 2], [3, 2, 3], [3]], [5, 9, 5]) == [False, True, True]
assert can_objects_fly([[1, 2], [3, 2, 3]], [5, 1]) == [False, False]
assert can_objects_fly([], []) == []
assert can_objects_fly([[1, 1], [2, 2, 2]], [2, 6]) == [True, True]