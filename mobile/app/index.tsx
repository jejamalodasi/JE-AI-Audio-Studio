import * as DocumentPicker from "expo-document-picker";
import { useAudioPlayer } from "expo-audio";
import { Directory, File, Paths } from "expo-file-system";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import {
  createSongJob,
  downloadArtifact,
  getSongJob,
  healthCheck,
} from "../src/api";
import { StudioControls } from "../src/studio-controls";
import { MixStudio } from "../src/mix-studio";
import {
  clearDraft,
  loadDraft,
  loadProjects,
  saveDraft,
  upsertProject,
  removeProject,
  type LocalProject,
} from "../src/localStore";
import type { PickedAudio, SongConfig, SongJob } from "../src/types";

const DEFAULT_CONFIG: SongConfig = {
  base_prompt:
    "Bengali folk-inspired emotional song, warm harmonium, bamboo flute, hand percussion, soft bass, organic modern production",
  cleanup_mode: "basic",
  sections: [
    { name: "Intro", duration_seconds: 4 },
    { name: "Verse", duration_seconds: 8 },
    { name: "Chorus", duration_seconds: 10 },
    { name: "Bridge", duration_seconds: 6 },
    { name: "Outro", duration_seconds: 6 },
  ],
  crossfade_seconds: 0.45,
  continuity: "vocal-anchor",
  guidance_scale: 3,
  temperature: 1,
  top_k: 250,
  top_p: 0,
  seed: 42,
  vocal_gain_db: -1,
  music_gain_db: -3,
  target_peak: 0.95,
  compression_ratio: 2,
  saturation: 0.08,
  device: "auto",
  max_total_seconds: 90,
};

function newProjectId() {
  return `local-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function projectTitle(file?: PickedAudio | null) {
  const name = file?.name || "Untitled AI Song";
  return name.replace(/\.[^/.]+$/, "") || "Untitled AI Song";
}

function Card({ children }: { children: ReactNode }) {
  return (
    <View
      style={{
        backgroundColor: "#141821",
        borderRadius: 20,
        padding: 18,
        gap: 12,
        borderWidth: 1,
        borderColor: "#252b38",
      }}
    >
      {children}
    </View>
  );
}

function ActionButton({
  label,
  onPress,
  disabled,
  secondary,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  secondary?: boolean;
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={{
        backgroundColor: secondary ? "#202633" : "#f4f6fa",
        borderRadius: 14,
        minHeight: 50,
        alignItems: "center",
        justifyContent: "center",
        paddingHorizontal: 18,
        opacity: disabled ? 0.45 : 1,
      }}
    >
      <Text
        selectable
        style={{
          color: secondary ? "#f4f6fa" : "#0b0d12",
          fontSize: 15,
          fontWeight: "800",
        }}
      >
        {label}
      </Text>
    </Pressable>
  );
}

function TinyButton({
  label,
  onPress,
  danger,
}: {
  label: string;
  onPress: () => void;
  danger?: boolean;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={{
        borderRadius: 10,
        borderWidth: 1,
        borderColor: danger ? "#5d3239" : "#303747",
        paddingHorizontal: 10,
        paddingVertical: 7,
      }}
    >
      <Text style={{ color: danger ? "#ff9b9b" : "#c7cfdd", fontSize: 11, fontWeight: "800" }}>
        {label}
      </Text>
    </Pressable>
  );
}

function AudioPreview({ uri }: { uri: string }) {
  const player = useAudioPlayer(uri);

  return (
    <Card>
      <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
        Final Master Preview
      </Text>
      <Text selectable style={{ color: "#8f98aa", fontSize: 13 }}>
        Saved locally on this device.
      </Text>
      <View style={{ flexDirection: "row", gap: 10 }}>
        <ActionButton label="▶ Play" onPress={() => player.play()} />
        <ActionButton label="⏸ Pause" onPress={() => player.pause()} secondary />
      </View>
    </Card>
  );
}

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const [file, setFile] = useState<PickedAudio | null>(null);
  const [config, setConfig] = useState<SongConfig>(DEFAULT_CONFIG);
  const [job, setJob] = useState<SongJob | null>(null);
  const [projects, setProjects] = useState<LocalProject[]>([]);
  const [localArtifacts, setLocalArtifacts] = useState<LocalProject["localArtifacts"]>({});
  const [projectId, setProjectId] = useState(newProjectId());
  const [createdAt, setCreatedAt] = useState(new Date().toISOString());
  const [message, setMessage] = useState("Select a vocal, then build your song.");
  const [busy, setBusy] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resumedJobId = useRef<string | null>(null);

  function projectSnapshot(
    nextJob: SongJob | undefined = job ?? undefined,
    nextArtifacts = localArtifacts,
  ): LocalProject {
    const now = new Date().toISOString();
    return {
      id: projectId,
      title: projectTitle(file),
      createdAt,
      updatedAt: now,
      source: file ?? undefined,
      config,
      job: nextJob,
      localArtifacts: nextArtifacts,
    };
  }

  async function persistHistory(
    nextJob: SongJob | undefined = job ?? undefined,
    nextArtifacts = localArtifacts,
  ) {
    const snapshot = projectSnapshot(nextJob, nextArtifacts);
    await upsertProject(snapshot);
    const nextProjects = await loadProjects();
    setProjects(nextProjects);
  }

  useEffect(() => {
    let active = true;

    async function restoreLocalState() {
      const [draft, savedProjects] = await Promise.all([loadDraft(), loadProjects()]);
      if (!active) return;

      setProjects(savedProjects);

      if (draft) {
        setProjectId(draft.id);
        setCreatedAt(draft.createdAt);
        setFile(draft.source ?? null);
        setConfig({
          ...DEFAULT_CONFIG,
          ...draft.config,
          sections: draft.config.sections?.length ? draft.config.sections : DEFAULT_CONFIG.sections,
          max_total_seconds: draft.config.max_total_seconds ?? DEFAULT_CONFIG.max_total_seconds,
        });
        setJob(draft.job ?? null);
        setLocalArtifacts(draft.localArtifacts ?? {});

        const sourceUri = draft.source?.uri;
        const sourceExists = sourceUri ? new File(sourceUri).exists : false;
        if (!draft.source || sourceExists) {
          setMessage("✅ Local draft restored.");
        } else {
          setFile(null);
          setMessage("Saved project restored, but its source file is no longer on the device.");
        }
      }

      setHydrated(true);
    }

    restoreLocalState().catch((error) => {
      if (active) {
        setHydrated(true);
        setMessage(error instanceof Error ? error.message : "Could not restore local projects.");
      }
    });

    return () => {
      active = false;
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

  useEffect(() => {
    if (!hydrated) return;

    const timer = setTimeout(() => {
      void saveDraft(projectSnapshot());
    }, 450);

    return () => clearTimeout(timer);
  }, [hydrated, file, config, job, localArtifacts, projectId, createdAt]);

  useEffect(() => {
    if (!hydrated || !job || busy) return;
    if (job.status !== "queued" && job.status !== "running") return;
    if (resumedJobId.current === job.job_id) return;

    resumedJobId.current = job.job_id;
    setBusy(true);
    pollJob(job.job_id);
  }, [hydrated, job?.job_id, job?.status, busy]);

  const totalDurationSeconds = config.sections.reduce(
    (sum, section) => sum + Math.max(0, section.duration_seconds),
    0,
  );
  const overDurationLimit = totalDurationSeconds > config.max_total_seconds;

  const completed = job?.status === "completed";
  const running = job?.status === "queued" || job?.status === "running";

  const statusTitle = useMemo(() => {
    if (!job) return "Ready";
    if (job.status === "queued") return "Queued";
    if (job.status === "running") return job.stage || "Processing";
    if (job.status === "failed") return "Failed";
    return "Complete";
  }, [job]);

  async function pickAudio() {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ["audio/*", "video/*"],
        copyToCacheDirectory: true,
        multiple: false,
      });

      if (result.canceled || !result.assets?.[0]) return;

      const asset = result.assets[0];
      const sourceDirectory = new Directory(Paths.document, "projects", "sources");
      sourceDirectory.create({ idempotent: true, intermediates: true });

      const safeName = (asset.name || "vocal.wav").replace(/[^a-zA-Z0-9._-]/g, "_");
      const persistentSource = new File(sourceDirectory, `${Date.now()}-${safeName}`);
      await new File(asset.uri).copy(persistentSource);

      const persistentFile: PickedAudio = {
        uri: persistentSource.uri,
        name: asset.name || "vocal.wav",
        mimeType: asset.mimeType,
      };

      const nextId = newProjectId();
      setProjectId(nextId);
      setCreatedAt(new Date().toISOString());
      setFile(persistentFile);
      setJob(null);
      setLocalArtifacts({});
      setMessage("✅ Reference copied to persistent app storage. Draft auto-save is on.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not select and save the file.");
    }
  }

  async function runHealth() {
    try {
      const result = await healthCheck();
      setMessage(`Connected: ${result.service} v${result.version}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "API health check failed.");
    }
  }

  function pollJob(jobId: string) {
    if (pollTimer.current) clearTimeout(pollTimer.current);

    const tick = async () => {
      try {
        const next = await getSongJob(jobId);
        setJob(next);
        void persistHistory(next, localArtifacts);

        if (next.status === "completed") {
          setBusy(false);
          setMessage("✅ Full AI Song is ready and the project history is updated.");
          return;
        }

        if (next.status === "failed") {
          setBusy(false);
          setMessage(next.error || "Generation failed.");
          return;
        }

        pollTimer.current = setTimeout(tick, 2500);
      } catch (error) {
        setBusy(false);
        setMessage(error instanceof Error ? error.message : "Could not read job status.");
      }
    };

    void tick();
  }

  async function buildSong() {
    if (!file) {
      setMessage("Choose a vocal/audio file first.");
      return;
    }
    if (overDurationLimit) {
      setMessage(`Reduce the arrangement to ${config.max_total_seconds} seconds or less before building.`);
      return;
    }

    setBusy(true);
    setJob(null);
    setLocalArtifacts({});
    setMessage("Uploading reference…");

    try {
      const created = await createSongJob(file, config);
      const createdJob: SongJob = {
        job_id: created.job_id,
        status: created.status === "running" ? "running" : "queued",
        progress: 0,
      };
      setJob(createdJob);
      await persistHistory(createdJob, {});
      setMessage(`Job ${created.job_id.slice(0, 8)} queued and saved locally…`);
      pollJob(created.job_id);
    } catch (error) {
      setBusy(false);
      setMessage(error instanceof Error ? error.message : "Could not create song job.");
    }
  }

  async function saveArtifact(
    artifact: "final" | "vocal" | "backing" | "bundle",
    extension: string,
  ) {
    if (!job?.job_id) return;

    try {
      setMessage(`Downloading ${artifact}…`);
      const uri = await downloadArtifact(job.job_id, artifact, extension);
      const nextArtifacts = { ...localArtifacts, [artifact]: uri };
      setLocalArtifacts(nextArtifacts);
      await persistHistory(job, nextArtifacts);
      if (artifact === "final") setMessage("✅ Final master saved to device.");
      else if (artifact === "bundle") setMessage("✅ Project ZIP saved to device.");
      else setMessage(`✅ ${artifact} audio saved to device.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Download failed.");
    }
  }

  async function openProject(saved: LocalProject) {
    setProjectId(saved.id);
    setCreatedAt(saved.createdAt);
    setConfig({
      ...DEFAULT_CONFIG,
      ...saved.config,
      sections: saved.config.sections?.length ? saved.config.sections : DEFAULT_CONFIG.sections,
      max_total_seconds: saved.config.max_total_seconds ?? DEFAULT_CONFIG.max_total_seconds,
    });
    setJob(saved.job ?? null);
    setLocalArtifacts(saved.localArtifacts ?? {});

    if (saved.source?.uri && new File(saved.source.uri).exists) {
      setFile(saved.source);
    } else {
      setFile(null);
    }

    const savedFinal = saved.localArtifacts?.final;
    setMessage(savedFinal && new File(savedFinal).exists ? "✅ Project opened with local master." : "✅ Project opened.");

    if (saved.job?.status === "queued" || saved.job?.status === "running") {
      setBusy(true);
      pollJob(saved.job.job_id);
    } else {
      setBusy(false);
    }
  }

  async function deleteSavedProject(id: string) {
    await removeProject(id);
    setProjects((current) => current.filter((project) => project.id !== id));
    if (id === projectId) {
      await clearDraft();
      setJob(null);
      setLocalArtifacts({});
      setFile(null);
      setConfig(DEFAULT_CONFIG);
      setProjectId(newProjectId());
      setCreatedAt(new Date().toISOString());
      setMessage("Local project removed from history and current draft.");
    }
  }

  return (
    <ScrollView
      contentInsetAdjustmentBehavior="automatic"
      contentContainerStyle={{
        paddingHorizontal: 16,
        paddingTop: 18,
        paddingBottom: Math.max(24, insets.bottom + 18),
        gap: 14,
      }}
    >
      <View style={{ gap: 5 }}>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 28, fontWeight: "900" }}>
          JE AI Audio Studio
        </Text>
        <Text selectable style={{ color: "#8f98aa", fontSize: 14, lineHeight: 20 }}>
          Vocal → AI arrangement → mix → master
        </Text>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginTop: 4 }}>
          <View
            style={{
              width: 8,
              height: 8,
              borderRadius: 8,
              backgroundColor: hydrated ? "#7ee787" : "#f5c26b",
            }}
          />
          <Text selectable style={{ color: "#aab2c0", fontSize: 12 }}>
            {hydrated ? "Local Save ON · No Login Required" : "Loading local projects…"}
          </Text>
        </View>
      </View>

      <Card>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
          01 · Source
        </Text>
        <Text selectable style={{ color: "#8f98aa" }}>
          {file ? file.name : "No vocal selected"}
        </Text>
        <ActionButton label="Choose Vocal / Audio" onPress={pickAudio} />
        <ActionButton label="Test API Connection" onPress={runHealth} secondary />
      </Card>

      <StudioControls config={config} setConfig={setConfig} />

      <Card>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
          03 · Build
        </Text>
        {running ? (
          <View style={{ gap: 10 }}>
            <ActivityIndicator />
            <Text selectable style={{ color: "#d8deea", fontSize: 14 }}>
              {statusTitle}
            </Text>
          </View>
        ) : (
          <ActionButton
            label={overDurationLimit ? "Fix Arrangement Length" : "🚀 Build Full AI Song"}
            onPress={buildSong}
            disabled={busy || overDurationLimit}
          />
        )}
        <Text selectable style={{ color: "#8f98aa", fontSize: 13, lineHeight: 19 }}>
          Heavy AI runs on the configured server/GPU backend. The Android app is the control surface.
        </Text>
      </Card>

      {job && (
        <Card>
          <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
            04 · Job Status
          </Text>
          <Text selectable style={{ color: "#c9d0dc", fontSize: 15, fontWeight: "700" }}>
            {statusTitle}
          </Text>
          {job.status === "queued" || job.status === "running" ? (
            <View style={{ gap: 7 }}>
              <View
                style={{
                  height: 8,
                  borderRadius: 8,
                  overflow: "hidden",
                  backgroundColor: "#252b38",
                }}
              >
                <View
                  style={{
                    width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%`,
                    height: "100%",
                    backgroundColor: "#f4f6fa",
                  }}
                />
              </View>
              <Text selectable style={{ color: "#8f98aa", fontSize: 12 }}>
                {Math.round(job.progress ?? 0)}%
              </Text>
            </View>
          ) : null}
          {job.error ? (
            <Text selectable style={{ color: "#ff8b8b", lineHeight: 20 }}>
              {job.error}
            </Text>
          ) : null}

          {completed ? (
            <View style={{ gap: 10 }}>
              <ActionButton label="⬇ Save Final WAV" onPress={() => void saveArtifact("final", "wav")} />
              <ActionButton label="⬇ Save Cleaned Vocal" onPress={() => void saveArtifact("vocal", "wav")} secondary />
              <ActionButton label="⬇ Save Backing" onPress={() => void saveArtifact("backing", "wav")} secondary />
              <ActionButton label="⬇ Save Project ZIP" onPress={() => void saveArtifact("bundle", "zip")} secondary />
              <Text selectable style={{ color: "#70798a", fontSize: 12, lineHeight: 18 }}>
                Files are saved under the app's persistent Documents storage so they are not treated as temporary cache.
              </Text>
            </View>
          ) : null}
        </Card>
      )}

      {localArtifacts?.final && new File(localArtifacts.final).exists ? (
        <AudioPreview uri={localArtifacts.final} />
      ) : null}

      {completed && job?.job_id ? (
        <MixStudio
          jobId={job.job_id}
          artifacts={localArtifacts}
          vocalGain={config.vocal_gain_db}
          backingGain={config.music_gain_db}
          targetPeak={config.target_peak}
          compressionRatio={config.compression_ratio}
          saturation={config.saturation}
          onRemixSaved={(uri) => {
            const nextArtifacts = { ...localArtifacts, remix: uri };
            setLocalArtifacts(nextArtifacts);
            void persistHistory(job, nextArtifacts);
          }}
          onMessage={setMessage}
        />
      ) : null}

      <Card>
        <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
          <View style={{ gap: 3, flex: 1 }}>
            <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
              05 · Local Project History
            </Text>
            <Text selectable style={{ color: "#7f8899", fontSize: 12 }}>
              Up to 25 recent projects are kept on this device.
            </Text>
          </View>
          <TinyButton
            label="Refresh"
            onPress={() => {
              void loadProjects().then(setProjects);
            }}
          />
        </View>

        {projects.length === 0 ? (
          <Text selectable style={{ color: "#8f98aa", fontSize: 13, lineHeight: 19 }}>
            No saved projects yet. Starting a build creates the first local project automatically.
          </Text>
        ) : (
          <View style={{ gap: 9 }}>
            {projects.slice(0, 8).map((saved) => (
              <View
                key={saved.id}
                style={{
                  borderWidth: 1,
                  borderColor: "#292f3c",
                  borderRadius: 14,
                  padding: 12,
                  gap: 8,
                }}
              >
                <View style={{ flexDirection: "row", justifyContent: "space-between", gap: 10 }}>
                  <View style={{ flex: 1, gap: 3 }}>
                    <Text selectable numberOfLines={1} style={{ color: "#dce2ec", fontWeight: "800" }}>
                      {saved.title}
                    </Text>
                    <Text selectable style={{ color: "#737d8f", fontSize: 11 }}>
                      {new Date(saved.updatedAt).toLocaleString()} · {saved.job?.status || "Draft"}
                    </Text>
                  </View>
                  <View style={{ flexDirection: "row", gap: 7 }}>
                    <TinyButton label="Open" onPress={() => void openProject(saved)} />
                    <TinyButton label="Delete" danger onPress={() => void deleteSavedProject(saved.id)} />
                  </View>
                </View>
              </View>
            ))}
          </View>
        )}
      </Card>

      <Text selectable style={{ color: "#70798a", fontSize: 12, lineHeight: 18 }}>
        {message}
      </Text>

      <Text selectable style={{ color: "#555e6e", fontSize: 11, lineHeight: 17 }}>
        Cloud account/login is not required for local saving. When multi-device sync is added later, an account will be used to associate cloud projects with you.
      </Text>

      <Text selectable style={{ color: "#555e6e", fontSize: 11, lineHeight: 17 }}>
        Prototype note: the current backend uses MusicGen Melody weights that are CC-BY-NC 4.0. Replace the backend before commercial deployment.
      </Text>
    </ScrollView>
  );
}
