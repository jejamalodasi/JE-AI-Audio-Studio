"""AI music/audio generation modules."""

from .basic_pitch_transcriber import transcribe_with_basic_pitch
from .vocal_to_melody import vocal_to_melody
from .vocal_to_music import VocalMusicConfig, generate_from_vocal

__all__ = [
    "transcribe_with_basic_pitch",
    "vocal_to_melody",
    "VocalMusicConfig",
    "generate_from_vocal",
]
