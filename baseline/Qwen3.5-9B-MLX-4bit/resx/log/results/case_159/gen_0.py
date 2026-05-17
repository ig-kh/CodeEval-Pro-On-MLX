
def eat(number, need, remaining):
    """
    You're a hungry rabbit, and you already have eaten a certain number of carrots,
    but now you need to eat more carrots to complete the day's meals.
    you should return an array of [ total number of eaten carrots after your meals,
                                    the number of carrots left after your meals ]
    if there are not enough remaining carrots, you will eat all remaining carrots, but will still be hungry.
    
    Example:
    * eat(5, 6, 10) -> [11, 4]
    * eat(4, 8, 9) -> [12, 1]
    * eat(1, 10, 10) -> [11, 0]
    * eat(2, 11, 5) -> [7, 0]
    
    Variables:
    @number : integer
        the number of carrots that you have eaten.
    @need : integer
        the number of carrots that you need to eat.
    @remaining : integer
        the number of remaining carrots thet exist in stock
    
    Constrain:
    * 0 <= number <= 1000
    * 0 <= need <= 1000
    * 0 <= remaining <= 1000

    Have fun :)
    """
    eaten = number + need
    if eaten > remaining:
        eaten = remaining
    return [eaten, remaining - eaten]

def feed_rabbits(rabbits, stock):
    """You are a farmer who needs to feed a group of hungry rabbits. Each rabbit has a specific number of carrots it has already eaten and a specific number it still needs to eat. You have a limited number of carrots in stock. Write a function that takes in a list of rabbits, where each rabbit is represented by a tuple (number, need), and the total number of carrots in stock. The function should return the total number of carrots eaten by all rabbits and the number of carrots left in stock after feeding all the rabbits.
    """
    total_eaten = 0
    remaining = stock
    
    for rabbit in rabbits:
        number, need = rabbit
        eaten, remaining = eat(number, need, remaining)
        total_eaten += eaten
        
    return [total_eaten, remaining]


assert feed_rabbits([(5, 6), (4, 8), (1, 10)], 25) == [24, 1]
assert feed_rabbits([(2, 11), (3, 5), (4, 7)], 20) == [20, 0]
assert feed_rabbits([(0, 5), (5, 5), (10, 5)], 30) == [15, 15]
assert feed_rabbits([(1, 10), (2, 11), (3, 12)], 50) == [33, 17]