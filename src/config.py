import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "gpt-4.1-mini"


def load_settings() -> Settings:
    # Read afresh on each explicit run; environment variables take precedence.
    values = dotenv_values(ENV_PATH)
    key = (os.environ.get("OPENAI_API_KEY", values.get("OPENAI_API_KEY")) or "").strip()
    if not key:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Add it to the repository .env file, "
            "then click Analyze documents again."
        )
    model = (os.environ.get("OPENAI_MODEL", values.get("OPENAI_MODEL")) or "gpt-4.1-mini").strip()
    return Settings(api_key=key, model=model or "gpt-4.1-mini")

