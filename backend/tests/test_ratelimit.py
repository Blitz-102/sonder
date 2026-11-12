import pytest

from sonder.sources.ratelimit import RateLimiter


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_first_call_does_not_wait() -> None:
    t = FakeTime()
    RateLimiter(0.2, clock=t.clock, sleep=t.sleep).wait()
    assert t.sleeps == []


def test_consecutive_calls_are_spaced() -> None:
    t = FakeTime()
    limiter = RateLimiter(3.0, clock=t.clock, sleep=t.sleep)
    for _ in range(3):
        limiter.wait()
    assert t.sleeps == [3.0, 3.0]


def test_only_remaining_interval_is_slept() -> None:
    t = FakeTime()
    limiter = RateLimiter(1.0, clock=t.clock, sleep=t.sleep)
    limiter.wait()
    t.now += 0.4
    limiter.wait()
    assert t.sleeps == pytest.approx([0.6])


def test_no_wait_after_interval_elapsed() -> None:
    t = FakeTime()
    limiter = RateLimiter(1.0, clock=t.clock, sleep=t.sleep)
    limiter.wait()
    t.now += 5
    limiter.wait()
    assert t.sleeps == []
