from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    lastfm_api_key: str | None = None
    data_dir: Path = Field(default=REPO_ROOT / "data", validation_alias="SONDER_DATA_DIR")

    @property
    def cache_db(self) -> Path:
        return self.data_dir / "cache.sqlite"

    @property
    def index_db(self) -> Path:
        return self.data_dir / "sonder.sqlite"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
