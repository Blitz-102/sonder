import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Enforces a minimum interval between consecutive calls to wait()."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._next_slot is not None and now < self._next_slot:
                self._sleep(self._next_slot - now)
                now = self._next_slot
            self._next_slot = now + self.min_interval


DEEZER_LIMITER = RateLimiter(0.2)
ITUNES_LIMITER = RateLimiter(3.0)
LASTFM_LIMITER = RateLimiter(1.0)
