"""
Authentication and authorization system
"""
from datetime import datetime, timedelta, timezone
import uuid
from typing import Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
import os
from app.services.refresh import RefreshTokenService

from app.models.sql_models import User
from app.core.database import get_db
from sqlalchemy.orm import Session
from app.core.config import settings


# Configuration
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token
security = HTTPBearer()


class AuthService:
    """Authentication service"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        data: Optional[dict] = None,
        subject: Optional[Union[str, uuid.UUID]] = None,
        expires_delta: Optional[timedelta] = None,
        extra_claims: Optional[dict] = None,
    ) -> str:
        """
        Backwards-compatible token creator.
        - old callers: create_access_token(data={"sub": username})
        - new callers: create_access_token(subject=user_id)
        """
        now = datetime.now(timezone.utc)
        if expires_delta is None:
            expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        exp = now + expires_delta

        payload: dict = {}
        # If caller provided a 'data' dict, merge it (backwards compat).
        if data:
            payload.update(data)

        # If caller passed a subject, overwrite/ensure 'sub'.
        if subject is not None:
            payload["sub"] = str(subject)

        # Merge extra claims if provided
        if extra_claims:
            payload.update(extra_claims)

        # Standard claims
        payload.setdefault("iat", int(now.timestamp()))
        payload.setdefault("type", "access")
        payload["exp"] = int(exp.timestamp())

        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        return token
    
    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            print("Decoded payload:", payload)
            return payload
        except JWTError:
            return None
        
    
    @staticmethod
    def create_refresh_token_in_db(db: Session, user: User, days_valid: int = 30) -> str:
        return RefreshTokenService.create_refresh_token(db, user, days_valid=days_valid)

    @staticmethod
    def rotate_refresh_token(db: Session, token_plain: str, days_valid: int = 30):
        return RefreshTokenService.verify_token_and_rotate(db, token_plain, rotate_days=days_valid)

    @staticmethod
    def revoke_refresh_token(db: Session, token_plain: str) -> bool:
        return RefreshTokenService.revoke_token(db, token_plain)

    @staticmethod
    def revoke_all_user_refresh_tokens(db: Session, user: User) -> int:
        return RefreshTokenService.revoke_all_tokens_for_user(db, user)


class UserService:
    """User management service"""
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username"""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def create_user(
        db: Session, 
        username: str, 
        email: str, 
        password: str, 
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user"""
        hashed_password = AuthService.get_password_hash(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = UserService.get_user_by_username(db, username)
        if not user:
            return None
        if not AuthService.verify_password(password, user.hashed_password):
            return None
        return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    print("Credentials:", credentials)
    
    try:
        payload = AuthService.verify_token(credentials.credentials)
        print("Token payload:", payload)
        if payload is None:
            raise credentials_exception
        userid: str = payload.get("sub")
        if userid is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = UserService.get_user_by_id(db, user_id=userid)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    print("Current user active status:", current_user )
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_superuser(current_user: User = Depends(get_current_active_user)) -> User:
    """Require superuser privileges"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

