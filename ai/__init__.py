"""AI music/audio generation and neural enhancement modules."""

from .arrangement_generator import AIConditionedArrangementConfig, generate_ai_conditioned_arrangement
from .basic_pitch_transcriber import transcribe_with_basic_pitch
from .conditioned_music import ConditionedMusicConfig, condition_vocal_to_music
from .musicgen_melody import MusicGenConfig, generate_musicgen_melody
from .neural_vocal_enhancement import enhance_vocal_neural
from .pitch_timing_ai import (
    PitchTimingAIConfig,
    analyze_pitch_neural,
    correct_pitch_timing_ai,
    correct_pitch_timing_file,
)
from .song_builder import SongBuilderConfig, SongSection, build_song_sketch
from .vocal_to_melody import vocal_to_melody
from .vocal_to_music import VocalMusicConfig, generate_from_vocal

__all__ = [
    "AIConditionedArrangementConfig",
    "generate_ai_conditioned_arrangement",
    "transcribe_with_basic_pitch",
    "ConditionedMusicConfig",
    "condition_vocal_to_music",
    "MusicGenConfig",
    "generate_musicgen_melody",
    "enhance_vocal_neural",
    "PitchTimingAIConfig",
    "analyze_pitch_neural",
    "correct_pitch_timing_ai",
    "correct_pitch_timing_file",
    "SongBuilderConfig",
    "SongSection",
    "build_song_sketch",
    "vocal_to_melody",
    "VocalMusicConfig",
    "generate_from_vocal",
]
