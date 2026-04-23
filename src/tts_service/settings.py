from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

InferenceMode = Literal["hybrid", "faster", "triton", "base"]
Dtype = Literal["bfloat16", "float16", "float32"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    omnivoice_mode: InferenceMode = "hybrid"
    omnivoice_dtype: Dtype = "bfloat16"
    cuda_device: int = 0
    # Local path to the OmniVoice snapshot (model files shipped with the image).
    # Pass `k2-fsa/OmniVoice` to download from HuggingFace at startup instead.
    omnivoice_model_path: str = "/app/models/OmniVoice"

    sample_rate_out: int = 8000
    max_text_length: int = 500

    voices_dir: Path = Path("./voices")
    voices_config: Path = Path("./voices/voices.json")

    queue_maxsize: int = 64
    request_timeout: float = 30.0

    mock_engine: bool = False
    prewarm: bool = True


settings = Settings()
