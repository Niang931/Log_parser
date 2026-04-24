import time
import threading

class TokenBucket:
    token_per_call = 10
    def __init__(self, max_capacity=100, refill_rate=10):
        self.max_capacity = max_capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self.tokens = max_capacity

    def lazy_refill(self):
        """Refill the tokens since the last refill no active background refilling"""
        current = time.time()
        time_elapsed = int(current - self.last_refill)
        self.last_refill = current
        self.tokens = min(self.max_capacity, self.tokens+ self.refill_rate * time_elapsed)

    def consume(self):
        self.lazy_refill()
        if self.tokens >= self.token_per_call:
            self.tokens -= self.token_per_call
        else:
            print("Request rejected")
