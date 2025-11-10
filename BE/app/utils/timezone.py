"""
Timezone utility functions for KST (Korea Standard Time)
"""
from datetime import datetime, timezone, timedelta

# KST는 UTC+9
KST = timezone(timedelta(hours=9))


def get_kst_now() -> datetime:
    """현재 KST 시간을 반환 (timezone-aware)"""
    return datetime.now(KST)


def get_kst_now_naive() -> datetime:
    """현재 KST 시간을 반환 (timezone-naive, DB 저장용)"""
    return datetime.now(KST).replace(tzinfo=None)


def utc_to_kst(utc_dt: datetime) -> datetime:
    """UTC datetime을 KST로 변환"""
    if utc_dt.tzinfo is None:
        # naive datetime은 UTC로 간주
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(KST)


def kst_to_utc(kst_dt: datetime) -> datetime:
    """KST datetime을 UTC로 변환"""
    if kst_dt.tzinfo is None:
        # naive datetime은 KST로 간주
        kst_dt = kst_dt.replace(tzinfo=KST)
    return kst_dt.astimezone(timezone.utc)

