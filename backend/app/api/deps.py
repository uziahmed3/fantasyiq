from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decode_access_token
from app.db.session import get_db
from app.models import User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/token", auto_error=False
)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession, token: Annotated[str | None, Depends(oauth2_scheme)] = None
) -> User:
    creds_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise creds_error
    payload = decode_access_token(token)
    if not payload or not payload.get("sub"):
        raise creds_error
    user = db.scalar(select(User).where(User.email == payload["sub"]))
    if user is None or not user.is_active:
        raise creds_error
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
