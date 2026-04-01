"""
Simple in-memory sliding-window rate limiter. No extra dependencies.
"""

from collections import defaultdict
from time import time
from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self):
        self._store: dict[str, list[float]] = defaultdict(list)

    def _ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, request: Request, limit: int, window: int = 60):
        ip = self._ip(request)
        now = time()
        self._store[ip] = [t for t in self._store[ip] if now - t < window]
        if len(self._store[ip]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {limit} requests per {window}s.",
            )
        self._store[ip].append(now)


rate_limiter = RateLimiter()
