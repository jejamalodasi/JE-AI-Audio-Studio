import { useState } from "react";
import * as Sharing from "expo-sharing";
import { Pressable, Text, View } from "react-native";

import { downloadArtifact, generateMusicParts } from "./api";
import type { SongConfig } from "./types";

type TrackId = "vocal" | "backing" | "melody" | "chords" | "bass" | "drums" | "rhythm";

export type TrackSetting = {
  volume: number;
  muted: boolean;
  solo: boolean;
};

export type TrackSettings = Partial<Record<TrackId, TrackSetting>>;

const TRACKS: Array<{ id: TrackId; label: string; kind: "audio" | "midi" }> = [
  { id: "vocal", label: "Vocal", kind: "audio" },
  { id: "backing", label: "AI Backing", kind: "audio" },
  { id: "melody", label: "Melody", kind: "midi" },
  { id: "chords", label: "Chords", kind: "midi" },
  { id: "bass", label: "Bass", kind: "midi" },
  { id: "drums", label: "Drums", kind: "midi" },
  { id: "rhythm", label: "Rhythm", kind: "midi" },
];

function defaultSetting(id: TrackId): TrackSetting {
  return {
    volume: id === "vocal" ? 0.9 : id === "backing" ? 0.75 : 1,
    muted: false,
    solo: false,
  };
}

function Button({
  label,
  onPress,
  disabled,
  active,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={{
        minHeight: 32,
        minWidth: 42,
        borderRadius: 9,
        paddingHorizontal: 9,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: active ? "#f4f6fa" : "#202633",
        borderWidth: 1,
        borderColor: active ? "#f4f6fa" : "#303747",
        opacity: disabled ? 0.35 : 1,
      }}
    >
      <Text selectable style={{ color: active ? "#0b0d12" : "#cbd3df", fontSize: 10, fontWeight: "900" }}>
        {label}
      </Text>
    </Pressable>
  );
}

export function TimelineStudio({
  jobId,
  config,
  artifacts,
  musicParts,
  trackSettings,
  onTrackSettingsChange,
  onMusicPartsChange,
  onMessage,
}: {
  jobId: string;
  config: SongConfig;
  artifacts: { vocal?: string; backing?: string };
  musicParts?: Partial<Record<"melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement", string>>;
  trackSettings: TrackSettings;
  onTrackSettingsChange: (settings: TrackSettings) => void;
  onMusicPartsChange: (
    parts: Partial<Record<"melody" | "chords" | "bass" | "drums" | "rhythm" | "arrangement", string>>,
  ) => void;
  onMessage: (message: string) => void;
}) {
  const [buildingParts, setBuildingParts] = useState(false);
  const totalSeconds = Math.max(
    1,
    config.sections.reduce((sum, section) => sum + Math.max(0, section.duration_seconds), 0),
  );
  const soloExists = TRACKS.some((track) => trackSettings[track.id]?.solo);

  function setting(id: TrackId): TrackSetting {
    return trackSettings[id] ?? defaultSetting(id);
  }

  function updateTrack(id: TrackId, patch: Partial<TrackSetting>) {
    onTrackSettingsChange({
      ...trackSettings,
      [id]: { ...setting(id), ...patch },
    });
  }

  async function createParts() {
    setBuildingParts(true);
    onMessage("Generating Melody, Chords, Bass, Drums and Rhythm MIDI parts…");
    try {
      const bpm = config.bpm ?? 120;
      const beatsPerBarSeconds = (60 / bpm) * 4;
      const bars = Math.max(1, Math.min(64, Math.ceil(totalSeconds / beatsPerBarSeconds)));
      const result = await generateMusicParts(jobId, {
        bpm,
        key: config.key ?? "C",
        scale: config.scale ?? "major",
        bars,
        seed: config.seed,
      });

      const downloaded: typeof result.parts = { ...result.parts };
      const entries = Object.keys(result.parts) as Array<keyof typeof result.parts>;
      for (const kind of entries) {
        downloaded[kind] = await downloadArtifact(jobId, kind, "mid");
      }

      onMusicPartsChange(downloaded);
      onMessage("✅ Real MIDI parts created and saved locally.");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Could not generate music parts.");
    } finally {
      setBuildingParts(false);
    }
  }

  async function shareMidi(uri: string, label: string) {
    try {
      if (!(await Sharing.isAvailableAsync())) {
        onMessage("Sharing is not available on this device.");
        return;
      }
      await Sharing.shareAsync(uri, {
        dialogTitle: "Share " + label + " MIDI",
        mimeType: "audio/midi",
      });
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Could not share MIDI.");
    }
  }

  return (
    <View
      style={{
        backgroundColor: "#141821",
        borderRadius: 20,
        padding: 18,
        gap: 14,
        borderWidth: 1,
        borderColor: "#252b38",
      }}
    >
      <View style={{ gap: 5 }}>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
          07 · Multi-Track Timeline
        </Text>
        <Text selectable style={{ color: "#7f8899", fontSize: 12, lineHeight: 18 }}>
          One timeline for the vocal, AI backing and generated musical parts. Audio monitor controls affect local preview; MIDI lanes are real exported parts.
        </Text>
      </View>

      <View style={{ gap: 5 }}>
        <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
          <Text selectable style={{ color: "#7f8899", fontSize: 11 }}>TIMELINE</Text>
          <Text selectable style={{ color: "#cbd3df", fontSize: 11, fontVariant: ["tabular-nums"] }}>
            {Math.round(totalSeconds)} sec
          </Text>
        </View>
        <View style={{ flexDirection: "row", gap: 3, height: 28 }}>
          {config.sections.map((section, index) => (
            <View
              key={section.name + "-" + index}
              style={{
                flex: Math.max(1, section.duration_seconds),
                borderRadius: 6,
                backgroundColor: "#202633",
                borderWidth: 1,
                borderColor: "#303747",
                alignItems: "center",
                justifyContent: "center",
                overflow: "hidden",
              }}
            >
              <Text selectable numberOfLines={1} style={{ color: "#aeb7c6", fontSize: 9, fontWeight: "800" }}>
                {section.name}
              </Text>
            </View>
          ))}
        </View>
      </View>

      <View style={{ gap: 8 }}>
        {TRACKS.map((track) => {
          const current = setting(track.id);
          const hasAudio = track.id === "vocal" ? Boolean(artifacts.vocal) : track.id === "backing" ? Boolean(artifacts.backing) : false;
          const midiUri = track.kind === "midi" ? musicParts?.[track.id as keyof typeof musicParts] : undefined;
          const available = hasAudio || Boolean(midiUri);

          return (
            <View
              key={track.id}
              style={{
                borderRadius: 13,
                backgroundColor: "#0e1117",
                borderWidth: 1,
                borderColor: current.solo ? "#626d81" : "#262e3a",
                padding: 11,
                gap: 8,
              }}
            >
              <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                <View style={{ width: 76, gap: 2 }}>
                  <Text selectable numberOfLines={1} style={{ color: "#e4e9f1", fontSize: 12, fontWeight: "800" }}>
                    {track.label}
                  </Text>
                  <Text selectable style={{ color: "#687284", fontSize: 9 }}>
                    {track.kind === "midi" ? "MIDI lane" : "Audio lane"}
                  </Text>
                </View>

                <View style={{ flex: 1, flexDirection: "row", gap: 3, height: 26 }}>
                  {config.sections.map((section, index) => (
                    <View
                      key={track.id + "-" + index}
                      style={{
                        flex: Math.max(1, section.duration_seconds),
                        borderRadius: 5,
                        backgroundColor: available ? "#252d3a" : "#171c25",
                        borderWidth: 1,
                        borderColor: available ? "#303747" : "#222832",
                        justifyContent: "center",
                        paddingHorizontal: 3,
                      }}
                    >
                      {index === 0 ? (
                        <Text selectable numberOfLines={1} style={{ color: available ? "#8e98a8" : "#4f5868", fontSize: 8 }}>
                          {available ? "CLIP" : "—"}
                        </Text>
                      ) : null}
                    </View>
                  ))}
                </View>

                <Text selectable style={{ color: "#7f8899", width: 40, textAlign: "right", fontSize: 9 }}>
                  {Math.round(current.volume * 100)}%
                </Text>
              </View>

              <View style={{ flexDirection: "row", gap: 6, flexWrap: "wrap" }}>
                <Button label="M" active={current.muted} onPress={() => updateTrack(track.id, { muted: !current.muted })} />
                <Button label="S" active={current.solo} onPress={() => updateTrack(track.id, { solo: !current.solo })} />
                <Button label="VOL −" disabled={current.volume <= 0} onPress={() => updateTrack(track.id, { volume: Number(Math.max(0, current.volume - 0.1).toFixed(2)) })} />
                <Button label="VOL +" disabled={current.volume >= 1} onPress={() => updateTrack(track.id, { volume: Number(Math.min(1, current.volume + 0.1).toFixed(2)) })} />
                {midiUri ? <Button label="Share MIDI" onPress={() => void shareMidi(midiUri, track.label)} /> : null}
              </View>
            </View>
          );
        })}
      </View>

      <Text selectable style={{ color: "#687284", fontSize: 11, lineHeight: 17 }}>
        {soloExists ? "Solo mode is active for at least one track; monitor intent is stored in the project." : "No solo track selected."}
      </Text>

      <Button
        label={buildingParts ? "Generating MIDI…" : musicParts?.arrangement ? "Regenerate MIDI Parts" : "Generate MIDI Parts"}
        disabled={buildingParts}
        onPress={() => void createParts()}
      />
    </View>
  );
}
