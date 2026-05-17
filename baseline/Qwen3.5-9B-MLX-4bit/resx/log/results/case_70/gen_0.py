
def strange_sort_list(lst):
    '''
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Examples:
    strange_sort_list([1, 2, 3, 4]) == [1, 4, 2, 3]
    strange_sort_list([5, 5, 5, 5]) == [5, 5, 5, 5]
    strange_sort_list([]) == []
    '''
    if not lst:
        return []
    
    sorted_lst = sorted(lst)
    result = []
    left, right = 0, len(sorted_lst) - 1
    
    while left <= right:
        if left == right:
            result.append(sorted_lst[left])
        else:
            # Alternate between taking from left (min) and right (max)
            result.append(sorted_lst[left])
            left += 1
            result.append(sorted_lst[right])
            right -= 1
            
    return result

def strange_sort_lists(lst_of_lsts):
    '''
    Given a list of lists of integers, return a list of lists in a strange order. 
    The strange order for each sublist should be calculated using the strange_sort_list function.
    Additionally, the final result should be sorted by the sum of the elements in each sublist in ascending order.
    '''
    if not lst_of_lsts:
        return []
    
    # Process each sublist with strange_sort_list
    processed_sublists = [strange_sort_list(sublist) for sublist in lst_of_lsts]
    
    # Calculate the sum of each processed sublist
    sums = [sum(sublist) for sublist in processed_sublists]
    
    # Combine the processed sublists and their sums into tuples for sorting
    combined = list(zip(processed_sublists, sums))
    
    # Sort by sum in ascending order
    combined.sort(key=lambda x: x[1])
    
    # Extract the sorted sublists from the combined list
    return [sublist for sublist, _ in combined]


assert strange_sort_lists([[1, 2, 3, 4], [5, 5, 5, 5], []]) == [[], [1, 4, 2, 3], [5, 5, 5, 5]]
assert strange_sort_lists([[10, 20], [1, 2, 3], [4, 5]]) == [[1, 3, 2], [4, 5], [10, 20]]
assert strange_sort_lists([[7, 7], [1, 2, 3, 4], [5]]) == [[5], [1, 4, 2, 3], [7, 7]]