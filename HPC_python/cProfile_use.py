# Created by LP
# Date: 2025-03-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

import cProfile
import time

def compute():
    total = 0
    for i in range(1000000):
        total += i ** 0.5  # Just something a bit heavy
    return total

if __name__ == "__main__":
    cProfile.run('compute()')