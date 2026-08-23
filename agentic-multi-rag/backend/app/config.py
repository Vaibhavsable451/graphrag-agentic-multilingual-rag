from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    gemini_api_key: str = ""

    primary_llm_provider: str = "groq"

    pinecone_api_key: str = ""
    pinecone_index_name: str = "agentic-multi-rag"
    pinecone_environment: str = "us-east-1"

    app_env: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
