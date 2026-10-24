from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    GROQ_API_KEY: str = ""
    WEBHOOK_SECRET: str = "change-me"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/data.db"
    CHROMA_PATH: str = "./data/chroma"
    GROQ_MODEL: str = "qwen/qwen3.8-27b"


settings = Settings()
