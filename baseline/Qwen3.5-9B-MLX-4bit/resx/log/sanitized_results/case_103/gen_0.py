
def rounded_avg(n, m):
    """You are given two positive integers n and m, and your task is to compute the
    average of the integers from n through m (including n and m). 
    Round the answer to the nearest integer and convert that to binary.
    If n is greater than m, return -1.
    Example:
    rounded_avg(1, 5) => "0b11"
    rounded_avg(7, 5) => -1
    rounded_avg(10, 20) => "0b1111"
    rounded_avg(20, 33) => "0b11010"
    """
    if n > m:
        return "-1"

    # Calculate the average of integers from n to m inclusive
    avg = (n + m) / 2

    # Round the average to the nearest integer
    rounded_avg = int(avg + 0.5)

    # Convert to binary and return with '0b' prefix
    return "0b" + bin(rounded_avg)[2:]

def process_tuples(tuples):
    """
    You are given a list of tuples, where each tuple contains two positive integers. Your task is to compute the rounded average for each tuple, convert the result to binary

assert process_tuples([(1, 5), (7, 5), (10, 20), (20, 33)]) == '11-1111111010'