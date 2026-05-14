from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    name: str
    email: str


class AuthService:
    def __init__(self, users_file: Path = USERS_FILE):
        self.users_file = users_file
        DATA_DIR.mkdir(exist_ok=True)
        if not self.users_file.exists():
            self._save_users([])

    def signup(self, name: str, email: str, password: str) -> AuthUser:
        clean_name = " ".join(name.strip().split())
        clean_email = email.strip().lower()
        self._validate_signup(clean_name, clean_email, password)

        users = self._load_users()
        if any(user["email"] == clean_email for user in users):
            raise ValueError("An account already exists for this email address.")

        user = {
            "user_id": str(uuid.uuid4()),
            "name": clean_name,
            "email": clean_email,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        users.append(user)
        self._save_users(users)
        return AuthUser(user_id=user["user_id"], name=user["name"], email=user["email"])

    def login(self, email: str, password: str) -> AuthUser:
        clean_email = email.strip().lower()
        for user in self._load_users():
            if user["email"] == clean_email and check_password_hash(user["password_hash"], password):
                return AuthUser(user_id=user["user_id"], name=user["name"], email=user["email"])
        raise ValueError("Invalid email or password.")

    def get_user(self, user_id: str | None) -> AuthUser | None:
        if not user_id:
            return None
        for user in self._load_users():
            if user["user_id"] == user_id:
                return AuthUser(user_id=user["user_id"], name=user["name"], email=user["email"])
        return None

    @staticmethod
    def _validate_signup(name: str, email: str, password: str) -> None:
        if len(name) < 2:
            raise ValueError("Enter your full name.")
        if not EMAIL_PATTERN.match(email):
            raise ValueError("Enter a valid email address.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

    def _load_users(self) -> list[dict]:
        try:
            payload = json.loads(self.users_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        users = payload.get("users", [])
        return users if isinstance(users, list) else []

    def _save_users(self, users: list[dict]) -> None:
        self.users_file.write_text(
            json.dumps({"users": users}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
