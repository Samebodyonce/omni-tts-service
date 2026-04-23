# Voices

The service registers voices from `voices.json`. Three modes are supported:

- **`auto`** — OmniVoice picks the speaker (deterministic at `temperature=0` for a given text). Safe default when you don't have reference audio yet.
- **`design`** — pass a comma-separated attribute string via `instruct` (e.g. `"male, low pitch"`). Only reliably trained on zh/en — results for kk/ru/tr may drift between inputs.
- **`clone`** — provide `ref_audio` (wav file placed in this directory) and optionally `ref_text` (transcription). Best consistency.

Once you have a reference recording for a language, edit the corresponding entry:

```json
{
  "id": "ru_female",
  "lang": "ru",
  "mode": "clone",
  "ref_audio": "ru_female.wav",
  "ref_text": "Здравствуйте, вы позвонили в колл-центр."
}
```

`ref_audio` is relative to this directory. If `ref_text` is omitted, OmniVoice will auto-transcribe via Whisper (slower startup).
