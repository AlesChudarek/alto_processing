from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Alto Processing Web"
    environment: str = "development"

    class Config:
        env_prefix = "ALTO_WEB_"


def get_settings() -> Settings:
    return Settings()
