import Storage from "expo-sqlite/kv-store";

import type { MusicPartFiles, PickedAudio, SongConfig, SongJob, TrackMixSettings } from "./types";

const DRAFT_KEY = "je-ai-audio-studio:draft:v1";
const PROJECTS_KEY = "je-ai-audio-studio:projects:v1";
const MAX_PROJECTS = 25;

export type LocalProject = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  source?: PickedAudio;
  config: SongConfig;
  job?: SongJob;
  localArtifacts?: {
    final?: string;
    vocal?: string;
    backing?: string;
    bundle?: string;
    remix?: string;
  };
  musicParts?: MusicPartFiles;
  trackMix?: TrackMixSettings;
};

async function readJson<T>(key: string, fallback: T): Promise<T> {
  try {
    const value = await Storage.getItem(key);
    if (!value) return fallback;
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

async function writeJson<T>(key: string, value: T): Promise<void> {
  await Storage.setItem(key, JSON.stringify(value));
}

export async function loadDraft(): Promise<LocalProject | null> {
  return readJson<LocalProject | null>(DRAFT_KEY, null);
}

export async function saveDraft(project: LocalProject): Promise<void> {
  await writeJson(DRAFT_KEY, project);
}

export async function clearDraft(): Promise<void> {
  await Storage.removeItem(DRAFT_KEY);
}

export async function loadProjects(): Promise<LocalProject[]> {
  const projects = await readJson<LocalProject[]>(PROJECTS_KEY, []);
  return Array.isArray(projects) ? projects : [];
}

export async function upsertProject(project: LocalProject): Promise<void> {
  const existing = await loadProjects();
  const withoutCurrent = existing.filter((item) => item.id !== project.id);
  const next = [project, ...withoutCurrent]
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, MAX_PROJECTS);
  await writeJson(PROJECTS_KEY, next);
}

export async function removeProject(projectId: string): Promise<void> {
  const existing = await loadProjects();
  await writeJson(
    PROJECTS_KEY,
    existing.filter((item) => item.id !== projectId),
  );
}

export async function clearProjects(): Promise<void> {
  await Storage.removeItem(PROJECTS_KEY);
}
