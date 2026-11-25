import tempfile
import unittest
from pathlib import Path

from request_rate_limiter import SharedRequestRateLimiter


class FakeTime:
    def __init__(self):
        self.now = 100.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SharedRequestRateLimiterTests(unittest.TestCase):
    def test_spacing_and_window_reset_are_shared(self):
        fake = FakeTime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            first = SharedRequestRateLimiter(
                state_path=path,
                max_requests=2,
                window_seconds=10,
                min_interval_seconds=2,
                clock=fake.clock,
                sleeper=fake.sleep,
            )
            second = SharedRequestRateLimiter(
                state_path=path,
                max_requests=2,
                window_seconds=10,
                min_interval_seconds=2,
                clock=fake.clock,
                sleeper=fake.sleep,
            )
            self.assertEqual(first.acquire()["window_count"], 1)
            reservation = second.acquire()
            self.assertEqual(reservation["window_count"], 2)
            self.assertEqual(reservation["waited_seconds"], 2)
            reservation = first.acquire()
            self.assertEqual(reservation["window_count"], 1)
            self.assertEqual(reservation["waited_seconds"], 8)


if __name__ == "__main__":
    unittest.main()
