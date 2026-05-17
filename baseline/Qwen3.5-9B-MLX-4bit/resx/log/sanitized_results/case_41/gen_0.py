

def car_race_collision(n: int):
    """
    Imagine a road that's a perfectly straight infinitely long line.
    n cars are driving left to right;  simultaneously, a different set of n cars
    are driving right to left.   The two sets of cars start out being very far from
    each other.  All cars move in the same speed.  Two cars are said to collide
    when a car that's moving left to right hits a car that's moving right to left.
    However, the cars are infinitely sturdy and strong; as a result, they continue moving
    in their trajectory as if they did not collide.

    This function outputs the number of such collisions.
    """
    return n

def car_race_collision_circular(n: int):
    """
    Imagine a circular racetrack with 2n cars, where n cars are driving clockwise and n cars are driving counterclockwise. The cars start at equal intervals around the racert


assert car_race_collision(3) + car_race_collision(3) == 6
assert car_race_collision(5) + car_race_collision(5) == 20
assert car_race_collision(10) + car_race_collision(10) == 90
assert car_race_collision(1) + car_race_collision(1) == 0
assert car_race_collision(0) + car_race_collision(0) == 0

