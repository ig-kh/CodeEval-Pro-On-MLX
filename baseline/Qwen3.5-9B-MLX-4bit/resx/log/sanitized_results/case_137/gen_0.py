
def compare_one(a, b):
    """
    Create a function that takes integers, floats, or strings representing
    real numbers, and returns the larger variable in its given variable type.
    Return None if the values are equal.
    Note: If a real number is represented as a string, the floating point might be . or ,

    compare_one(1, 2.5) ➞ 2.5
    compare_one(1, "2,3") ➞ "2,3"
    compare_one("5,1", "6") ➞ "6"
    compare_one("1", 1) ➞ None
    """
    import re

    def parse_value(val):
        if isinstance(val, (int, float)):
            return val
        else:
            # Replace comma with dot and parse as float
            return float(re.sub(r',', '.', val))

    def get_type(val):
        if isinstance(val, (int, float)):
            return "numeric"
        else:
            return "string"

    val_a = parse_value(a)
    val_b = parse_value(b)

    if val_a == val_b:
        return None

    if get_type(a) == "numeric" and get_type(b) == "numeric":
        return max(a, b)
    elif get_type(a) == "string" and get_type(b) == "string":
        return max(str(a), str(b))
    else:
        # One is numeric, the other is string
        if val_a > val_b:
            return a
        else:
            return b

def compare_many(pairs):
    """
    Create a function that takes a list of pairs of integers, floats, or strings representing real numbers, and returns a list of the larger variables in their given variable type for each pair. If the values in a pair are equal, the corresponding element in the result list should be None. Note: If a

assert compare_many([(1, 2.5), (1, '2,3'), ('5,1', '6'), ('1', 1)]) == [2.5, '2,3', '6', None]
assert compare_many([(3, 3), ('4', '4'), (5.5, 5.5)]) == [None, None, None]
assert compare_many([('7', 8), (9.9, '9,9'), (10, '10,1')]) == [8, None, '10,1']