from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth
import os
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User

# Initialize Firebase Admin SDK
# Note: You need to set FIREBASE_CREDENTIALS_PATH in .env
# and download your Firebase service account key

security = HTTPBearer(auto_error=False)

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


def get_or_create_user(db: Session, firebase_uid: str, email: str, name: str) -> User:
    """
    Get or create user in database based on Firebase UID
    """
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    
    if not user:
        # Create new user
        user = User(
            firebase_uid=firebase_uid,
            name=name or email.split('@')[0] if email else "Unknown User"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Created new user in DB: {firebase_uid}")
    else:
        # Update existing user info if changed
        updated_name = name or email.split('@')[0] if email else user.name
        if user.name != updated_name:
            user.name = updated_name
            db.commit()
            db.refresh(user)
    
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> dict:
    """
    Verify Firebase ID token, sync with DB, and return user info
    """
    # Check if Firebase is initialized
    if not firebase_admin._apps:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firebase Admin SDK not initialized. Please configure Firebase credentials.",
        )
    
    # Check if credentials are provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        
        firebase_uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name") or decoded_token.get("display_name") or (email.split('@')[0] if email else "Unknown User")
        
        # Sync user with database
        db_user = get_or_create_user(db, firebase_uid, email, name)
        
        return {
            "uid": firebase_uid,
            "db_user_id": db_user.id,  # DB user ID 추가
            "email": email,
            "name": name,
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
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[dict]:
    """
    Optional authentication - returns None if no credentials provided
    """
    if credentials is None:
        return None
    return await get_current_user(credentials, db)


async def get_current_user_dev(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> dict:
    """
    Development mode authentication - returns mock user if no credentials or Firebase not configured
    """
    # Check if Firebase is initialized
    if not firebase_admin._apps:
        # Firebase not configured, return mock user for development
        return {
            "uid": "dev-user-001",
            "db_user_id": None,
            "email": "dev@example.com",
            "name": "Development User",
            "picture": None
        }

    # If credentials provided, verify them
    if credentials:
        try:
            return await get_current_user(credentials, db)
        except Exception as e:
            # If token verification fails, fallback to mock user
            print(f"Token verification failed: {e}, using mock user")
            return {
                "uid": "dev-user-001",
                "db_user_id": None,
                "email": "dev@example.com",
                "name": "Development User",
                "picture": None
            }

    # No credentials provided, return mock user
    return {
        "uid": "dev-user-001",
        "db_user_id": None,
        "email": "dev@example.com",
        "name": "Development User",
        "picture": None
    }
