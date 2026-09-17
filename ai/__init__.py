"""AI music/audio generation and neural enhancement modules."""

from .basic_pitch_transcriber import transcribe_with_basic_pitch
from .neural_vocal_enhancement import enhance_vocal_neural
from .vocal_to_melody import vocal_to_melody
from .vocal_to_music import VocalMusicConfig, generate_from_vocal

__all__ = [
    "transcribe_with_basic_pitch",
    "enhance_vocal_neural",
    "vocal_to_melody",
    "VocalMusicConfig",
    "generate_from_vocal",
]
