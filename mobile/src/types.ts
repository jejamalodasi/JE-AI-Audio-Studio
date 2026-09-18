export type PickedAudio = {
  uri: string;
  name: string;
  mimeType?: string;
};

export type SongJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stage?: string;
  progress?: number;
  error?: string;
  artifacts?: {
    final: string;
    vocal: string;
    backing: string;
    bundle: string;
    remix?: string;
  };
};

export type TrackMixState = {
  volume: number;
  muted: boolean;
  solo: boolean;
};

export type TrackMixSettings = Partial<Record<
  "vocal" | "backing" | "melody" | "chords" | "bass" | "drums" | "rhythm",
  TrackMixState
>>;

export type SongConfig = {
  base_prompt: string;
  cleanup_mode: "none" | "basic" | "advanced";
  bpm?: number | null;
  key?: string | null;
  scale?: string | null;
  sections: Array<{ name: string; duration_seconds: number }>;
  crossfade_seconds: number;
  continuity: "vocal-anchor" | "chain";
  guidance_scale: number;
  temperature: number;
  top_k: number;
  top_p: number;
  seed: number;
  vocal_gain_db: number;
  music_gain_db: number;
  target_peak: number;
  compression_ratio: number;
  saturation: number;
  device: "auto" | "cpu" | "cuda";
  max_total_seconds: number;
};

export type MusicPartFiles = Partial<Record<
  "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement",
  string
>>;

export type TimelineClipKind = "audio" | "midi";

export type TimelineClip = {
  id: string;
  trackId: "vocal" | "backing" | "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement";
  kind: TimelineClipKind;
  label: string;
  startSec: number;
  durationSec: number;
  sourceOffsetSec: number;
  sourceUri?: string;
  sourceArtifact?: "vocal" | "backing";
  sourcePart?: "melody" | "chords" | "bass" | "drums" | "rhythm";
  fadeInSec: number;
  fadeOutSec: number;
};

export type MidiNote = {
  note: number;
  velocity: number;
  startBeat: number;
  durationBeat: number;
  trackName?: string;
};

export type TimelineState = {
  clips: TimelineClip[];
  playheadSec: number;
  zoom: number;
};

export type TimelineRenderResult = {
  job_id: string;
  status: "completed";
  filename: string;
};
