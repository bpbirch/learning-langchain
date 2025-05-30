from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    model_config = SettingsConfigDict(
        extra="allow",
        env_file=".env"
    )


settings: Settings = Settings()
