
def pluck(arr):
    """
    "Given an array representing a branch of a tree that has non-negative integer nodes
    your task is to pluck one of the nodes and return it.
    The plucked node should be the node with the smallest even value.
    If multiple nodes with the same smallest even value are found return the node that has smallest index.

    The plucked node should be returned in a list, [ smalest_value, its index ],
    If there are no even values or the given array is empty, return [].

    Example 1:
        Input: [4,2,3]
        Output: [2, 1]
        Explanation: 2 has the smallest even value, and 2 has the smallest index.

    Example 2:
        Input: [1,2,3]
        Output: [2, 1]
        Explanation: 2 has the smallest even value, and 2 has the smallest index. 

    Example 3:
        Input: []
        Output: []
    
    Example 4:
        Input: [5, 0, 3, 0, 4, 2]
        Output: [0, 1]
        Explanation: 0 is the smallest value, but  there are two zeros,
                     so we will choose the first zero, which has the smallest index.

    Constraints:
        * 1 <= nodes.length <= 10000
        * 0 <= node.value
    """
    even_nodes = [(val, idx) for idx, val in enumerate(arr) if val % 2 == 0]
    if not even_nodes:
        return []
    return [min(even_nodes, key=lambda x: (x[0], x[1]))]

def pluck_branches(branches):
    """pluck_branches is a function that takes a list of branches from a tree, where each branch is represented as an array of non-negative integer nodes, and returns a list of plucked nodes.
    >>> pluck_branches([[4,2,3], [1,2,3], [], [5, 0, 3, 0, 4, 2]])
    [[2, 1], [2, 1], [], [0, 1]]
    >>> pluck_branches([[4,2,3], [1,2,3], [], [5, 0, 3, 0, 4, 2]])
    [[2, 1], [2, 1], [], [0, 

assert pluck_branches([[4,2,3], [1,2,3], [], [5, 0, 3, 0, 4, 2]]) == [[2, 1], [2, 1], [], [0, 1]]
assert pluck_branches([[1, 3, 5], [2, 4, 6], [7, 8, 9]]) == [[], [2, 0], [8, 1]]
assert pluck_branches([[0, 2, 4], [1, 3, 5], [6, 8, 10]]) == [[0, 0], [], [6, 0]]
assert pluck_branches([[1], [2], [3], [4]]) == [[], [2, 0], [], [4, 0]]