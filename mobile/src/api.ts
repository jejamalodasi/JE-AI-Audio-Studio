import { Directory, File, Paths } from "expo-file-system";
import type { PickedAudio, SongConfig, SongJob } from "./types";

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

export function artifactUrl(jobId: string, artifact: "final" | "vocal" | "backing" | "bundle") {
  requireApiUrl();
  return `${API_URL}/api/jobs/${encodeURIComponent(jobId)}/download/${artifact}`;
}

export async function downloadArtifact(
  jobId: string,
  artifact: "final" | "vocal" | "backing" | "bundle",
  extension: string,
) {
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
