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
  };
};

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
};
