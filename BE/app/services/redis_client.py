import redis
import json
from typing import Optional, Any
from app.core.config import settings

class RedisClient:
    """Redis client for caching"""

    def __init__(self):
        self.client = None
        try:
            # Try to connect to Redis
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            self.client = redis.from_url(redis_url, decode_responses=True)
            self.client.ping()
            print("Redis connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            print("Caching will be disabled")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration (default 1 hour)"""
        if not self.client:
            return False

        try:
            self.client.setex(key, expire, json.dumps(value))
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.client:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    def invalidate_pattern(self, pattern: str) -> bool:
        """Invalidate all keys matching pattern"""
        if not self.client:
            return False

        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            print(f"Redis invalidate error: {e}")
            return False


# Global Redis client instance
redis_client = RedisClient()
