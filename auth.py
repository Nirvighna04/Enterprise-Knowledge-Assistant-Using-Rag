"""
auth.py — Authentication helpers: register and login (no OTP).
"""

import hashlib
from db import add_user, get_user_by_email


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


# ── Register ──────────────────────────────────────────────────────────────────

def register_user(email: str, password: str):
    """
    Returns:
        (True,  "ok")      — registered successfully
        (False, "exists")  — email already taken
    """
    if get_user_by_email(email):
        return False, "exists"

    add_user(email, hash_password(password))
    return True, "ok"


# ── Login ─────────────────────────────────────────────────────────────────────

def login_user(email: str, password: str):
    """
    Returns:
        (True,  user)        — credentials valid
        (False, "no_user")   — email not found
        (False, "bad_pass")  — wrong password
    """
    user = get_user_by_email(email)
    if not user:
        return False, "no_user"

    if not check_password(password, user.password_hash):
        return False, "bad_pass"

    return True, user
