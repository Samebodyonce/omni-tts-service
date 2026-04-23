import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .schemas import Lang, VoiceInfo, VoiceMode

log = logging.getLogger(__name__)


@dataclass
class Voice:
    id: str
    lang: Lang
    mode: VoiceMode
    description: str | None = None
    # clone mode — omnivoice-triton wants an on-disk path
    ref_audio_path: Path | None = None
    ref_text: str | None = None
    # design mode
    instruct: str | None = None

    def validate(self) -> None:
        if self.mode == "clone":
            if self.ref_audio_path is None:
                raise ValueError(f"voice {self.id}: clone mode requires ref_audio")
            if not self.ref_audio_path.exists():
                raise FileNotFoundError(
                    f"voice {self.id}: ref audio {self.ref_audio_path} not found"
                )
        elif self.mode == "design" and not self.instruct:
            raise ValueError(f"voice {self.id}: design mode requires instruct")

    def info(self) -> VoiceInfo:
        return VoiceInfo(id=self.id, lang=self.lang, mode=self.mode, description=self.description)


DEFAULT_VOICES: list[dict] = [
    {
        "id": "kk_default",
        "lang": "kk",
        "mode": "auto",
        "description": "Kazakh — auto voice (replace with cloned voice once ref audio is available)",
    },
    {
        "id": "ru_default",
        "lang": "ru",
        "mode": "auto",
        "description": "Russian — auto voice (replace with cloned voice once ref audio is available)",
    },
    {
        "id": "tr_default",
        "lang": "tr",
        "mode": "auto",
        "description": "Turkish — auto voice (replace with cloned voice once ref audio is available)",
    },
]


class VoiceRegistry:
    def __init__(self) -> None:
        self._voices: dict[str, Voice] = {}

    def load(self, config_path: Path, voices_dir: Path) -> None:
        raw: list[dict]
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            log.info("loaded %d voices from %s", len(raw), config_path)
        else:
            raw = DEFAULT_VOICES
            log.warning("voices config %s not found — using built-in defaults (auto voice)", config_path)

        for entry in raw:
            voice = Voice(
                id=entry["id"],
                lang=entry["lang"],
                mode=entry["mode"],
                description=entry.get("description"),
                ref_audio_path=(voices_dir / entry["ref_audio"]) if entry.get("ref_audio") else None,
                ref_text=entry.get("ref_text"),
                instruct=entry.get("instruct"),
            )
            voice.validate()
            self._voices[voice.id] = voice

    def get(self, voice_id: str) -> Voice:
        try:
            return self._voices[voice_id]
        except KeyError as e:
            raise KeyError(f"unknown voice id: {voice_id}") from e

    def list(self) -> list[VoiceInfo]:
        return [v.info() for v in self._voices.values()]
