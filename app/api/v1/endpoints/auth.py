from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status,Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, Any
import uuid

from app.core.auth import (
    AuthService,
    UserService,
    get_current_active_user,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from app.core.database import get_db
from app.models.sql_models import User

router = APIRouter(prefix="/auth", tags=["authentication"])


# ---------- Models ----------

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool

    model_config = {"from_attributes": True,
                    "protected_namespaces": ()}

    @field_validator("id", mode="before")
    def convert_uuid_to_str(cls, v: Any) -> str:
        if isinstance(v, uuid.UUID):
            return str(v)
        return v


class TokenWithRefresh(BaseModel):
    access_token: str
    token_type: str
    refresh_token: str
    user: UserResponse

class RefreshRequest(BaseModel):
    refresh_token: str

class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: str
    


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str
    
class LogoutRequest(BaseModel):
    refresh_token: str


# ---------- Existing endpoints (small improvements) ----------

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    if UserService.get_user_by_username(db, user_data.username):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")

    if UserService.get_user_by_email(db, user_data.email):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = UserService.create_user(
        db=db,
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name,
    )

    return UserResponse.from_orm(user)


# --- login (update) ---
@router.post("/login", response_model=TokenWithRefresh)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = UserService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # access token - put subject as user.id (stable)
    access_token = AuthService.create_access_token(subject=str(user.id))
    # refresh token stored in DB (opaque)
    refresh_token_plain = AuthService.create_refresh_token_in_db(db, user, days_valid=30)

    return TokenWithRefresh(
        access_token=access_token,
        token_type="bearer",
        refresh_token=refresh_token_plain,
        user=UserResponse.from_orm(user),
    )

# @router.post("/refresh", response_model=TokenRefreshResponse)
# async def refresh_token(req: TokenRefreshRequest):
#     """
#     Exchange a refresh token for a new access token.
#     Requires AuthService.verify_refresh_token() implementation.
#     """
#     payload = AuthService.verify_refresh_token(req.refresh_token)
#     if not payload:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

#     userid = payload.get("sub") or payload.get("username") or payload.get("user")
#     access_token = AuthService.create_access_token(data={"sub": userid}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
#     return TokenRefreshResponse(access_token=access_token)

@router.post("/refresh")
def refresh_tokens(request: RefreshRequest, db: Session = Depends(get_db)):
    payload = AuthService.rotate_refresh_token(db, request.refresh_token)
    if payload is None:
        raise HTTPException(401, "Invalid or revoked refresh token")

    user, new_refresh_plain = payload
    access_token = AuthService.create_access_token(subject=str(user.id))
    # Return new pair
    return {
    "access_token": access_token,
    "token_type": "bearer",
    "refresh_token": new_refresh_plain
    }

@router.post("/logout")
def logout(request: LogoutRequest, db: Session = Depends(get_db)):
    success = AuthService.revoke_refresh_token(db, request.refresh_token)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid refresh token")
    return {"msg": "Logged out"}

@router.post("/logout-all")
def logout_all(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    count = AuthService.revoke_all_user_refresh_tokens(db, user)
    return {"msg": f"Revoked {count} tokens"}

# --- refresh endpoint ---
@router.post("/refresh", response_model=RefreshResponse)
def refresh_token(req: RefreshRequest = Body(...), db: Session = Depends(get_db)):
    """
    Verify refresh token, rotate it, and return new access + refresh tokens.
    """
    result = AuthService.rotate_refresh_token(db, req.refresh_token)
    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked refresh token")

    user, new_refresh_plain = result
    access_token = AuthService.create_access_token(subject=str(user.id))
    return RefreshResponse(access_token=access_token, refresh_token=new_refresh_plain)

# --- logout (revoke current refresh token) ---
class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: LogoutRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Logout: revoke provided refresh token. If none provided, revoke all tokens for current user.
    """
    if payload.refresh_token:
        ok = AuthService.revoke_refresh_token(db, payload.refresh_token)
        if not ok:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid refresh token")
        return None

    # no token provided -> revoke all user's tokens
    AuthService.revoke_all_user_refresh_tokens(db, current_user)
    return None# --- refresh endpoint ---
@router.post("/refresh", response_model=RefreshResponse)
def refresh_token(req: RefreshRequest = Body(...), db: Session = Depends(get_db)):
    """
    Verify refresh token, rotate it, and return new access + refresh tokens.
    """
    result = AuthService.rotate_refresh_token(db, req.refresh_token)
    if result is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked refresh token")

    user, new_refresh_plain = result
    access_token = AuthService.create_access_token(subject=str(user.id))
    return RefreshResponse(access_token=access_token, refresh_token=new_refresh_plain)

# --- logout (revoke current refresh token) ---
class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    payload: LogoutRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Logout: revoke provided refresh token. If none provided, revoke all tokens for current user.
    """
    if payload.refresh_token:
        ok = AuthService.revoke_refresh_token(db, payload.refresh_token)
        if not ok:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid refresh token")
        return None

    # no token provided -> revoke all user's tokens
    AuthService.revoke_all_user_refresh_tokens(db, current_user)
    return None

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    print("Current user:", current_user)
    """Get current user information"""
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UpdateUserRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update current user information"""
    update_data = user_update.model_dump(exclude_unset=True)
    # Use UserService to perform safe updates and validation
    updated = UserService.update_user(db=db, user=current_user, updates=update_data)
    return UserResponse.from_orm(updated)


@router.post("/me/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(payload: ChangePasswordRequest, current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Change password for authenticated user"""
    ok = UserService.change_password(db=db, user=current_user, old_password=payload.old_password, new_password=payload.new_password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password incorrect")
    return None


@router.post("/password-reset/request", status_code=status.HTTP_202_ACCEPTED)
async def password_reset_request(payload: PasswordResetRequest, db: Session = Depends(get_db)):
    """
    Generates a password reset token and sends email.
    Requires UserService.create_password_reset_token + Email sending.
    """
    user = UserService.get_user_by_email(db, payload.email)
    if not user:
        # For privacy, respond accepted even if email not found
        return None

    token = UserService.create_password_reset_token(db, user)
    # send token by email (implement EmailService.send_password_reset)
    AuthService.send_password_reset_email(email=payload.email, token=token)
    return None


@router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
async def password_reset_confirm(payload: PasswordResetConfirm, db: Session = Depends(get_db)):
    """
    Confirm password reset using token + new password.
    Requires UserService.verify_password_reset_token + UserService.set_password.
    """
    user = UserService.get_user_by_password_reset_token(db, payload.token)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token")
    UserService.set_password(db, user, payload.new_password)
    return None


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_account(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    """Delete current user's account"""
    UserService.delete_user(db, current_user)
    return None
