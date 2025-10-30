"""
Redis caching utility for FastAPI
"""
import json
import functools
from typing import Optional, Any, Callable
from app.core.config import settings

try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None


class CacheManager:
    """Manages Redis caching with fallback to in-memory cache"""

    def __init__(self):
        self.redis_client: Optional[Redis] = None
        self.memory_cache: dict = {}
        self.enabled = settings.ENABLE_CACHE and REDIS_AVAILABLE

        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # Test connection
                self.redis_client.ping()
                print("✓ Redis cache connected successfully")
            except Exception as e:
                print(f"⚠ Redis not available, using in-memory cache: {e}")
                self.redis_client = None
                self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled:
            return self.memory_cache.get(key)

        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                print(f"Cache get error: {e}")

        return self.memory_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        if not self.enabled:
            self.memory_cache[key] = value
            return True

        ttl = ttl or settings.CACHE_TTL

        if self.redis_client:
            try:
                serialized = json.dumps(value, default=str)
                self.redis_client.setex(key, ttl, serialized)
                return True
            except Exception as e:
                print(f"Cache set error: {e}")

        self.memory_cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled:
            self.memory_cache.pop(key, None)
            return True

        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"Cache delete error: {e}")

        self.memory_cache.pop(key, None)
        return True

    def clear_pattern(self, pattern: str) -> bool:
        """Clear all keys matching pattern"""
        if not self.enabled:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.memory_cache[key]
            return True

        if self.redis_client:
            try:
                keys = self.redis_client.keys(f"*{pattern}*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                print(f"Cache clear pattern error: {e}")

        return True

    def flush_all(self) -> bool:
        """Clear all cache"""
        if not self.enabled:
            self.memory_cache.clear()
            return True

        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"Cache flush error: {e}")

        self.memory_cache.clear()
        return True


# Singleton instance
cache_manager = CacheManager()


def cached(
    key_prefix: str = "",
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Decorator to cache function results

    Usage:
        @cached(key_prefix="datasets", ttl=300)
        async def get_datasets():
            return await db.query(Dataset).all()
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder
                arg_str = "_".join(str(arg) for arg in args)
                kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                print(f"✓ Cache hit: {cache_key}")
                return cached_result

            # Execute function
            print(f"⊗ Cache miss: {cache_key}")
            result = await func(*args, **kwargs)

            # Store in cache
            cache_manager.set(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


def invalidate_cache(pattern: str):
    """
    Invalidate cache by pattern

    Usage:
        invalidate_cache("datasets")
    """
    cache_manager.clear_pattern(pattern)
