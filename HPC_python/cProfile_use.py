import cProfile
import time

def compute():
    total = 0
    for i in range(1000000):
        total += i ** 0.5  # Just something a bit heavy
    return total

if __name__ == "__main__":
    cProfile.run('compute()')