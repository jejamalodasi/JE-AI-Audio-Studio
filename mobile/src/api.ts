import { Directory, File, Paths } from "expo-file-system";
import type {
  MidiNote,
  PickedAudio,
  SongConfig,
  SongJob,
  TimelineRenderResult,
  TimelineState,
  TrackMixSettings,
} from "./types";

const API_URL = (process.env.EXPO_PUBLIC_API_URL || "").replace(/\/$/, "");

function requireApiUrl() {
  if (!API_URL) throw new Error("EXPO_PUBLIC_API_URL is not configured.");
}

async function parseError(response: Response) {
  const body = await response.text().catch(() => "");
  let detail = body;
  try {
    const json = JSON.parse(body);
    detail = json.detail || json.message || body;
  } catch {}
  return new Error(`HTTP ${response.status}: ${detail || "Request failed"}`);
}

export async function healthCheck() {
  requireApiUrl();
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<{ ok: boolean; service: string; version: string }>;
}

export async function createSongJob(file: PickedAudio, config: SongConfig) {
  requireApiUrl();
  const form = new FormData();
  form.append("file", {
    uri: file.uri,
    name: file.name,
    type: file.mimeType || "audio/wav",
  } as never);
  form.append("config", JSON.stringify(config));

  const response = await fetch(`${API_URL}/api/jobs/song`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw await parseError(response);
  return response.json() as Promise<{ job_id: string; status: string; poll: string }>;
}

export async function getSongJob(jobId: string): Promise<SongJob> {
  requireApiUrl();
  const response = await fetch(`${API_URL}/api/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export type ArtifactKind =
  | "timeline"
  | "final"
  | "vocal"
  | "backing"
  | "bundle"
  | "remix"
  | "melody"
  | "chords"
  | "bass"
  | "drums"
  | "rhythm"
  | "arrangement";

export function artifactUrl(jobId: string, artifact: ArtifactKind) {
  requireApiUrl();
  return `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/download/${artifact}`;
}

export async function downloadArtifact(
  jobId: string,
  artifact: ArtifactKind,
  extension: string,
) {
  requireApiUrl();
  const exportDirectory = new Directory(Paths.document, "exports");
  exportDirectory.create({ idempotent: true, intermediates: true });

  const filename = `je_ai_${artifact}_${Date.now()}.${extension}`;
  const destination = new File(exportDirectory, filename);
  const downloaded = await File.downloadFileAsync(
    artifactUrl(jobId, artifact),
    destination,
    { idempotent: true },
  );
  return downloaded.uri;
}

export type RemixResult = {
  job_id: string;
  status: "completed";
  remix_path: string;
  filename: string;
};

export async function remixSongJob(
  jobId: string,
  values: {
    vocal_gain_db: number;
    backing_gain_db: number;
    target_peak: number;
    compression_ratio: number;
    saturation: number;
  },
): Promise<RemixResult> {
  requireApiUrl();
  const form = new FormData();
  form.append("vocal_gain_db", String(values.vocal_gain_db));
  form.append("backing_gain_db", String(values.backing_gain_db));
  form.append("target_peak", String(values.target_peak));
  form.append("compression_ratio", String(values.compression_ratio));
  form.append("saturation", String(values.saturation));

  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/remix`,
    {
      method: "POST",
      body: form,
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export type MusicPartsResult = {
  job_id: string;
  status: "completed";
  parts: Partial<Record<"melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement", string>>;
  metadata: Record<string, unknown>;
};

export async function generateMusicParts(
  jobId: string,
  values: { bpm: number; key: string; scale: string; bars: number; seed: number },
): Promise<MusicPartsResult> {
  requireApiUrl();
  const form = new FormData();
  form.append("bpm", String(values.bpm));
  form.append("key", values.key);
  form.append("scale", values.scale);
  form.append("bars", String(values.bars));
  form.append("seed", String(values.seed));

  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/parts`,
    {
      method: "POST",
      body: form,
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export type ClipEditAction = "clean" | "fix-pitch" | "fix-timing";

export type ClipEditResult = {
  job_id: string;
  edit_id: string;
  status: "completed";
  action: ClipEditAction;
  source: "vocal" | "backing";
  download: string;
};

export async function editSongClip(
  jobId: string,
  values: {
    source: "vocal" | "backing";
    action: ClipEditAction;
    strength?: number;
  },
): Promise<ClipEditResult> {
  requireApiUrl();
  const form = new FormData();
  form.append("source", values.source);
  form.append("action", values.action);
  form.append("strength", String(values.strength ?? 0.65));

  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/clip-edit`,
    {
      method: "POST",
      body: form,
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function downloadMidiEdit(
  jobId: string,
  part: "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement",
  extension = "mid",
) {
  requireApiUrl();
  const exportDirectory = new Directory(Paths.document, "exports");
  exportDirectory.create({ idempotent: true, intermediates: true });

  const filename = `je_ai_midi_${encodeURIComponent(part)}_${Date.now()}.${extension}`;
  const destination = new File(exportDirectory, filename);
  const downloaded = await File.downloadFileAsync(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/download/midi-${encodeURIComponent(part)}`,
    destination,
    { idempotent: true },
  );
  return downloaded.uri;
}

export async function downloadClipEdit(
  jobId: string,
  editId: string,
  extension = "wav",
) {
  requireApiUrl();
  const exportDirectory = new Directory(Paths.document, "exports");
  exportDirectory.create({ idempotent: true, intermediates: true });

  const filename = `je_ai_clip_${encodeURIComponent(editId)}_${Date.now()}.${extension}`;
  const destination = new File(exportDirectory, filename);
  const downloaded = await File.downloadFileAsync(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/clip-edits/${encodeURIComponent(editId)}`,
    destination,
    { idempotent: true },
  );
  return downloaded.uri;
}


export async function getMidiNotes(
  jobId: string,
  part: "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement",
): Promise<{ part: string; ticksPerBeat: number; notes: MidiNote[] }> {
  requireApiUrl();
  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/midi/${encodeURIComponent(part)}`,
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}


export type MidiEditResult = {
  job_id: string;
  part: "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement";
  filename: string;
  download: string;
};

export async function saveMidiEdits(
  jobId: string,
  values: {
    part: "melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement";
    notes: Array<{
      note: number;
      velocity: number;
      startBeat: number;
      durationBeat: number;
    }>;
    bpm: number;
  },
): Promise<MidiEditResult> {
  requireApiUrl();
  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/midi-edit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}

export async function renderTimeline(
  jobId: string,
  values: {
    timeline: TimelineState;
    trackMix: TrackMixSettings;
    targetPeak: number;
    compressionRatio: number;
    saturation: number;
  },
): Promise<TimelineRenderResult> {
  requireApiUrl();
  const response = await fetch(
    `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/render-timeline`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    },
  );
  if (!response.ok) throw await parseError(response);
  return response.json();
}
