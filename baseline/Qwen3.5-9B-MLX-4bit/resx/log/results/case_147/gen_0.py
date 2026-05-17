
def get_max_triples(n):
    """
    You are given a positive integer n. You have to create an integer array a of length n.
        For each i (1 ≤ i ≤ n), the value of a[i] = i * i - i + 1.
        Return the number of triples (a[i], a[j], a[k]) of a where i < j < k, 
    and a[i] + a[j] + a[k] is a multiple of 3.

    Example :
        Input: n = 5
        Output: 1
        Explanation: 
        a = [1, 3, 7, 13, 21]
        The only valid triple is (1, 7, 13).
    """
    a = [i * i - i + 1 for i in range(1, n + 1)]
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (a[i] + a[j] + a[k]) % 3 == 0:
                    count += 1
    
    return count

def get_max_quadruples(nums):
    """
    Given a list of positive integers, you need to find the number of valid quadruples (a[i], a[j], a[k], a[l]) where i < j < k < l, and the sum of the elements in each quadruple is a multiple of 3. To solve this problem, you can use the function `get_max_triples` to find the number of valid triples for each subset of three elements in the list, and then combine these results to find the number of valid quadruples.
    """
    count = 0
    n = len(nums)
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 

assert get_max_quadruples([1, 3, 7, 13, 21]) == 1
assert get_max_quadruples([1, 2, 3, 4, 5]) == 0
assert get_max_quadruples([3, 6, 9, 12, 15]) == 5