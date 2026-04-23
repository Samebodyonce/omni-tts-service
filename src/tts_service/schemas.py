from typing import Literal

from pydantic import BaseModel, Field

Lang = Literal["kk", "ru", "tr"]
VoiceMode = Literal["clone", "design", "auto"]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    lang: Lang = Field(..., description="Language code")
    voice: str = Field(..., description="Voice id registered on the server")


class VoiceInfo(BaseModel):
    id: str
    lang: Lang
    mode: VoiceMode
    description: str | None = None


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    mode: str
    mock: bool
    queue_size: int
    queue_maxsize: int
