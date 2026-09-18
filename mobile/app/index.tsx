import * as DocumentPicker from "expo-document-picker";
import { useAudioPlayer } from "expo-audio";
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
};

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

function AudioPreview({ uri }: { uri: string }) {
  const player = useAudioPlayer(uri);

  return (
    <View style={{ gap: 10 }}>
      <Text selectable style={{ color: "#aab2c0", fontSize: 13 }}>
        Local final master preview
      </Text>
      <View style={{ flexDirection: "row", gap: 10 }}>
        <ActionButton label="▶ Play" onPress={() => player.play()} />
        <ActionButton label="⏸ Pause" onPress={() => player.pause()} secondary />
      </View>
    </View>
  );
}

export default function HomeScreen() {
  const insets = useSafeAreaInsets();
  const [file, setFile] = useState<PickedAudio | null>(null);
  const [config, setConfig] = useState<SongConfig>(DEFAULT_CONFIG);
  const [job, setJob] = useState<SongJob | null>(null);
  const [message, setMessage] = useState("Select a vocal, then build your song.");
  const [busy, setBusy] = useState(false);
  const [downloadedUri, setDownloadedUri] = useState<string | null>(null);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (pollTimer.current) clearTimeout(pollTimer.current);
    };
  }, []);

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
      setFile({
        uri: asset.uri,
        name: asset.name || "vocal.wav",
        mimeType: asset.mimeType,
      });
      setJob(null);
      setDownloadedUri(null);
      setMessage("Reference loaded. Ready for generation.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not select the file.");
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

        if (next.status === "completed") {
          setBusy(false);
          setMessage("✅ Full AI Song is ready.");
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

    tick();
  }

  async function buildSong() {
    if (!file) {
      setMessage("Choose a vocal/audio file first.");
      return;
    }

    setBusy(true);
    setJob(null);
    setDownloadedUri(null);
    setMessage("Uploading reference…");

    try {
      const created = await createSongJob(file, config);
      setMessage(`Job ${created.job_id.slice(0, 8)} queued…`);
      pollJob(created.job_id);
    } catch (error) {
      setBusy(false);
      setMessage(error instanceof Error ? error.message : "Could not create song job.");
    }
  }

  async function downloadFinal() {
    if (!job?.job_id) return;
    try {
      setMessage("Downloading final master…");
      const uri = await downloadArtifact(job.job_id, "final", "wav");
      setDownloadedUri(uri);
      setMessage("✅ Final master downloaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Download failed.");
    }
  }

  async function downloadBundle() {
    if (!job?.job_id) return;
    try {
      setMessage("Downloading project bundle…");
      const uri = await downloadArtifact(job.job_id, "bundle", "zip");
      setDownloadedUri(uri);
      setMessage("✅ Project ZIP downloaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Bundle download failed.");
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

      <Card>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
          02 · AI Song Direction
        </Text>
        <TextInput
          multiline
          value={config.base_prompt}
          onChangeText={(base_prompt) => setConfig((current) => ({ ...current, base_prompt }))}
          placeholder="Describe the style…"
          placeholderTextColor="#626c7d"
          style={{
            minHeight: 100,
            borderRadius: 14,
            backgroundColor: "#0e1117",
            color: "#f5f7fb",
            padding: 14,
            textAlignVertical: "top",
            borderWidth: 1,
            borderColor: "#242b38",
          }}
        />
        <Text selectable style={{ color: "#9da6b5", fontSize: 13, lineHeight: 19 }}>
          Structure: Intro 4s · Verse 8s · Chorus 10s · Bridge 6s · Outro 6s
        </Text>
      </Card>

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
          <ActionButton label="🚀 Build Full AI Song" onPress={buildSong} disabled={busy} />
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
          {job.error ? (
            <Text selectable style={{ color: "#ff8b8b", lineHeight: 20 }}>
              {job.error}
            </Text>
          ) : null}

          {completed ? (
            <View style={{ gap: 10 }}>
              <ActionButton label="⬇ Download Final WAV" onPress={downloadFinal} />
              <ActionButton label="⬇ Download Project ZIP" onPress={downloadBundle} secondary />
              <Text selectable style={{ color: "#70798a", fontSize: 12, lineHeight: 18 }}>
                Download the final WAV first to unlock local playback.
              </Text>
            </View>
          ) : null}
        </Card>
      )}

      {downloadedUri?.endsWith(".wav") ? <AudioPreview uri={downloadedUri} /> : null}

      <Text selectable style={{ color: "#70798a", fontSize: 12, lineHeight: 18 }}>
        {message}
      </Text>

      <Text selectable style={{ color: "#555e6e", fontSize: 11, lineHeight: 17 }}>
        Prototype note: the current backend uses MusicGen Melody weights that are CC-BY-NC 4.0. Replace the backend before commercial deployment.
      </Text>
    </ScrollView>
  );
}
