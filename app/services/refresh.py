# app/core/refresh_service.py

from datetime import datetime
from sqlalchemy.orm import Session
from app.models.sql_models import RefreshToken, User
from app.utils.token_utils import generate_refresh_token_string, hash_token, refresh_token_expiry
from typing import Optional, Tuple

class RefreshTokenService:
    @staticmethod
    def create_refresh_token(db: Session, user: User, days_valid: int = 30) -> str:
        """
        Create an opaque refresh token, store its hash in DB and return the plain token.
        """
        token = generate_refresh_token_string()
        token_hash = hash_token(token)
        expires_at = refresh_token_expiry(days=days_valid)

        rt = RefreshToken(
            token_hash=token_hash,
            user_id=str(user.id),
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            revoked=False,
            replaced_by=None,
        )
        db.add(rt)
        db.commit()
        db.refresh(rt)
        return token  # plain (only returned to client)

    @staticmethod
    def get_by_hash(db: Session, token_hash: str) -> Optional[RefreshToken]:
        return db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()

    @staticmethod
    def verify_token_and_rotate(db: Session, token_plain: str, rotate_days: int = 30) -> Optional[Tuple[User, str]]:
        """
        Verify incoming refresh token:
          - If valid + not revoked + not expired => rotate: create new refresh token,
            set old.revoked=True and old.replaced_by=new.id, return (user, new_plain_token).
          - If token already revoked => return None (optionally revoke whole session).
        """
        token_hash = hash_token(token_plain)
        rt = RefreshTokenService.get_by_hash(db, token_hash)
        if rt is None:
            return None
        if rt.revoked:
            return None
        if rt.expires_at < datetime.utcnow():
            return None

        # rotate: issue new token, mark old revoked and link
        user = db.query(User).filter(User.id == rt.user_id).first()
        if user is None:
            return None

        new_token_plain = generate_refresh_token_string()
        new_token_hash = hash_token(new_token_plain)
        new_expires_at = refresh_token_expiry(days=rotate_days)

        new_rt = RefreshToken(
            token_hash=new_token_hash,
            user_id=rt.user_id,
            created_at=datetime.utcnow(),
            expires_at=new_expires_at,
            revoked=False,
            replaced_by=None,
        )
        db.add(new_rt)

        # revoke old
        rt.revoked = True
        rt.replaced_by = new_rt.id

        db.commit()
        db.refresh(new_rt)
        return user, new_token_plain

    @staticmethod
    def revoke_token(db: Session, token_plain: str) -> bool:
        token_hash = hash_token(token_plain)
        rt = RefreshTokenService.get_by_hash(db, token_hash)
        if not rt:
            return False
        rt.revoked = True
        db.add(rt)
        db.commit()
        return True

    @staticmethod
    def revoke_all_tokens_for_user(db: Session, user: User) -> int:
        tokens = db.query(RefreshToken).filter(RefreshToken.user_id == str(user.id), RefreshToken.revoked == False).all()
        count = 0
        for t in tokens:
            t.revoked = True
            count += 1
        db.commit()
        return count
