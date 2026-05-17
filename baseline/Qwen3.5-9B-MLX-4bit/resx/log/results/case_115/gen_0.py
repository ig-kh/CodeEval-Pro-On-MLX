
def max_fill(grid, capacity):
    import math
    """
    You are given a rectangular grid of wells. Each row represents a single well,
    and each 1 in a row represents a single unit of water.
    Each well has a corresponding bucket that can be used to extract water from it, 
    and all buckets have the same capacity.
    Your task is to use the buckets to empty the wells.
    Output the number of times you need to lower the buckets.

    Example 1:
        Input: 
            grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]
            bucket_capacity : 1
        Output: 6

    Example 2:
        Input: 
            grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]
            bucket_capacity : 2
        Output: 5
    
    Example 3:
        Input: 
            grid : [[0,0,0], [0,0,0]]
            bucket_capacity : 5
        Output: 0

    Constraints:
        * all wells have the same length
        * 1 <= grid.length <= 10^2
        * 1 <= grid[:,1].length <= 10^2
        * grid[i][j] -> 0 | 1
        * 1 <= capacity <= 10
    """
    total_drops = 0
    
    for well in grid:
        # Count the number of water units (1s) in the current well
        water_units = sum(well)
        
        # Calculate the number of drops needed for this well
        drops = math.ceil(water_units / capacity)
        
        total_drops += drops
    
    return total_drops

def total_bucket_lowers(grids, capacities):
    """You are given a list of rectangular grids of wells, each with a corresponding bucket capacity. Each grid represents a different set of wells, and each well has a corresponding bucket that can be

assert total_bucket_lowers([[[0,0,1,0], [0,1,0,0], [1,1,1,1]], [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]], [1, 2]) == 11
assert total_bucket_lowers([[[0,0,0], [0,0,0]], [[1,1,1], [1,1,1]]], [5, 1]) == 6
assert total_bucket_lowers([[[1,1,1], [1,1,1]], [[0,0,0], [0,0,0]]], [1, 5]) == 6
assert total_bucket_lowers([[[1,0,1], [0,1,0]], [[1,1,1], [1,1,1]]], [2, 3]) == 4
assert total_bucket_lowers([[[0,0,0], [0,0,0]], [[0,0,0], [0,0,0]]], [5, 5]) == 0