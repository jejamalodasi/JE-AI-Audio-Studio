"""AI music/audio generation modules."""

from .vocal_to_melody import vocal_to_melody
from .vocal_to_music import VocalMusicConfig, generate_from_vocal

__all__ = ["vocal_to_melody", "VocalMusicConfig", "generate_from_vocal"]
