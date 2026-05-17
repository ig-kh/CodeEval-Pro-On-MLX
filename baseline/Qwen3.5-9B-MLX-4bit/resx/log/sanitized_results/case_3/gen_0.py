from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    import math
    import re
    from typing import List

    balance = 0

    for op in operations:
        balance += op

        if balance < 0:
            return True

    return False


def below_zero_multi_account(operations: List[List[int]]) -> bool:
    """ You are given a list of transactions where each transaction is a list of deposit and withdrawal operations on multiple bank accounts. Each sublist represents the operations for a single account,

assert any_account_below_zero([[1, 2, 3], [4, 5, 6]]) == False
assert any_account_below_zero([[1, 2, -4, 5], [1, 2, 3]]) == True
assert any_account_below_zero([[1, 2, 3], [1, -2, 3]]) == True
assert any_account_below_zero([[1, 2, 3], [1, 2, 3]]) == False