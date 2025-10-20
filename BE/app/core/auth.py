from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth
import os

# Initialize Firebase Admin SDK
# Note: You need to set FIREBASE_CREDENTIALS_PATH in .env
# and download your Firebase service account key

security = HTTPBearer()

def init_firebase():
    """Initialize Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("Firebase Admin SDK initialized successfully")
            else:
                print(f"Warning: Firebase credentials file not found at {cred_path}")
                print("Firebase authentication will not work. Please add your firebase-credentials.json")
    except Exception as e:
        print(f"Error initializing Firebase: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify Firebase ID token and return user info
    """
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "name": decoded_token.get("name"),
            "picture": decoded_token.get("picture")
        }
    except firebase_admin.exceptions.FirebaseError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """
    Optional authentication - returns None if no credentials provided
    """
    if credentials is None:
        return None
    return await get_current_user(credentials)


async def get_current_user_dev(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> dict:
    """
    Development mode authentication - returns mock user if no credentials or Firebase not configured
    """
    # Check if Firebase is initialized
    if not firebase_admin._apps:
        # Firebase not configured, return mock user for development
        return {
            "uid": "dev-user-001",
            "email": "dev@example.com",
            "name": "Development User",
            "picture": None
        }

    # If credentials provided, verify them
    if credentials:
        try:
            token = credentials.credentials
            decoded_token = auth.verify_id_token(token)
            return {
                "uid": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "name": decoded_token.get("name"),
                "picture": decoded_token.get("picture")
            }
        except Exception as e:
            # If token verification fails, fallback to mock user
            print(f"Token verification failed: {e}, using mock user")
            return {
                "uid": "dev-user-001",
                "email": "dev@example.com",
                "name": "Development User",
                "picture": None
            }

    # No credentials provided, return mock user
    return {
        "uid": "dev-user-001",
        "email": "dev@example.com",
        "name": "Development User",
        "picture": None
    }
