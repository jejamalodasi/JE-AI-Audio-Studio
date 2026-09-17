"""AI music/audio generation and neural enhancement modules."""

from .basic_pitch_transcriber import transcribe_with_basic_pitch
from .conditioned_music import ConditionedMusicConfig, condition_vocal_to_music
from .neural_vocal_enhancement import enhance_vocal_neural
from .pitch_timing_ai import (
    PitchTimingAIConfig,
    analyze_pitch_neural,
    correct_pitch_timing_ai,
    correct_pitch_timing_file,
)
from .vocal_to_melody import vocal_to_melody
from .vocal_to_music import VocalMusicConfig, generate_from_vocal

__all__ = [
    "transcribe_with_basic_pitch",
    "ConditionedMusicConfig",
    "condition_vocal_to_music",
    "enhance_vocal_neural",
    "PitchTimingAIConfig",
    "analyze_pitch_neural",
    "correct_pitch_timing_ai",
    "correct_pitch_timing_file",
    "vocal_to_melody",
    "VocalMusicConfig",
    "generate_from_vocal",
]
