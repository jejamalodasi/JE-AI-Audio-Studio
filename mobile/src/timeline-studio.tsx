import { useEffect, useMemo, useRef, useState } from "react";
import * as Sharing from "expo-sharing";
import { Pressable, ScrollView, Text, View } from "react-native";

import { downloadArtifact, generateMusicParts } from "./api";
import type {
  MusicPartFiles,
  SongConfig,
  TimelineClip,
  TimelineState,
  TrackMixSettings,
} from "./types";

type TrackId = TimelineClip["trackId"];

type TrackDefinition = {
  id: TrackId;
  label: string;
  kind: "audio" | "midi";
};

const TRACKS: TrackDefinition[] = [
  { id: "vocal", label: "Vocal", kind: "audio" },
  { id: "backing", label: "AI Backing", kind: "audio" },
  { id: "melody", label: "Melody", kind: "midi" },
  { id: "chords", label: "Chords", kind: "midi" },
  { id: "bass", label: "Bass", kind: "midi" },
  { id: "drums", label: "Drums", kind: "midi" },
  { id: "rhythm", label: "Rhythm", kind: "midi" },
];

const LABEL_W = 92;
const RULER_H = 30;
const TRACK_H = 62;
const MIN_ZOOM = 28;
const MAX_ZOOM = 92;
const ZOOM_STEP = 8;
const EDIT_STEP = 0.5;

function defaultSetting(id: TrackId): TrackMixSettings[TrackId] extends infer T ? T : never {
  return {
    volume: id === "vocal" ? 0.9 : id === "backing" ? 0.75 : 1,
    muted: false,
    solo: false,
  } as never;
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
        minHeight: 34,
        minWidth: 42,
        borderRadius: 9,
        paddingHorizontal: 10,
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

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function totalSeconds(config: SongConfig) {
  return Math.max(1, config.sections.reduce((sum, section) => sum + Math.max(0, section.duration_seconds), 0));
}

function defaultTimeline(
  config: SongConfig,
  artifacts: { vocal?: string; backing?: string },
  musicParts?: MusicPartFiles,
): TimelineState {
  const total = totalSeconds(config);
  const audioClips: TimelineClip[] = [];
  if (artifacts.vocal) {
    audioClips.push({
      id: "clip-vocal-main",
      trackId: "vocal",
      kind: "audio",
      label: "Vocal",
      startSec: 0,
      durationSec: total,
      sourceUri: artifacts.vocal,
      sourceArtifact: "vocal",
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }
  if (artifacts.backing) {
    audioClips.push({
      id: "clip-backing-main",
      trackId: "backing",
      kind: "audio",
      label: "AI Backing",
      startSec: 0,
      durationSec: total,
      sourceUri: artifacts.backing,
      sourceArtifact: "backing",
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }

  for (const track of TRACKS.filter((item) => item.kind === "midi")) {
    const uri = musicParts?.[track.id as keyof MusicPartFiles];
    if (!uri) continue;
    audioClips.push({
      id: `clip-${track.id}-main`,
      trackId: track.id,
      kind: "midi",
      label: track.label,
      startSec: 0,
      durationSec: total,
      sourceUri: uri,
      sourcePart: track.id,
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }

  return { clips: audioClips, playheadSec: 0, zoom: 48 };
}

export function TimelineStudio({
  jobId,
  config,
  artifacts,
  musicParts,
  trackSettings,
  initialTimeline,
  onTimelineChange,
  onTrackSettingsChange,
  onMusicPartsChange,
  onMessage,
}: {
  jobId: string;
  config: SongConfig;
  artifacts: { vocal?: string; backing?: string };
  musicParts?: MusicPartFiles;
  trackSettings: TrackMixSettings;
  initialTimeline?: TimelineState;
  onTimelineChange: (timeline: TimelineState) => void;
  onTrackSettingsChange: (settings: TrackMixSettings) => void;
  onMusicPartsChange: (parts: MusicPartFiles) => void;
  onMessage: (message: string) => void;
}) {
  const [timeline, setTimeline] = useState<TimelineState>(
    () => initialTimeline ?? defaultTimeline(config, artifacts, musicParts),
  );
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [buildingParts, setBuildingParts] = useState(false);
  const rulerRef = useRef<ScrollView | null>(null);

  useEffect(() => {
    if (initialTimeline) {
      setTimeline(initialTimeline);
    }
  }, [initialTimeline]);

  useEffect(() => {
    onTimelineChange(timeline);
  }, [timeline, onTimelineChange]);

  const total = totalSeconds(config);
  const timelineWidth = Math.max(420, total * timeline.zoom);
  const selectedClip = timeline.clips.find((clip) => clip.id === selectedClipId) ?? null;
  const soloExists = TRACKS.some((track) => trackSettings[track.id]?.solo);

  const sectionMarkers = useMemo(() => {
    let cursor = 0;
    return config.sections.map((section, index) => {
      const start = cursor;
      cursor += Math.max(0, section.duration_seconds);
      return { ...section, index, start };
    });
  }, [config.sections]);

  function updateTimeline(mutator: (current: TimelineState) => TimelineState) {
    setTimeline((current) => {
      const next = mutator(current);
      return {
        ...next,
        playheadSec: clamp(next.playheadSec, 0, total),
        zoom: clamp(next.zoom, MIN_ZOOM, MAX_ZOOM),
      };
    });
  }

  function setPlayhead(seconds: number) {
    updateTimeline((current) => ({ ...current, playheadSec: seconds }));
  }

  function snapTime(seconds: number) {
    return clamp(Math.round(seconds / EDIT_STEP) * EDIT_STEP, 0, total);
  }

  function clipForTrack(id: TrackId) {
    return timeline.clips.filter((clip) => clip.trackId === id);
  }

  function selectAt(seconds: number, trackId?: TrackId) {
    if (trackId) {
      const candidate = clipForTrack(trackId).find(
        (clip) => seconds >= clip.startSec && seconds <= clip.startSec + clip.durationSec,
      );
      setSelectedClipId(candidate?.id ?? null);
    }
    setPlayhead(seconds);
  }

  function updateClip(id: string, patch: Partial<TimelineClip>) {
    updateTimeline((current) => ({
      ...current,
      clips: current.clips.map((clip) => (clip.id === id ? { ...clip, ...patch } : clip)),
    }));
  }

  function splitSelected() {
    if (!selectedClip) return;
    const splitAt = snapTime(timeline.playheadSec);
    const clipEnd = selectedClip.startSec + selectedClip.durationSec;
    if (splitAt <= selectedClip.startSec + EDIT_STEP || splitAt >= clipEnd - EDIT_STEP) {
      onMessage("Move the playhead inside the selected clip before splitting.");
      return;
    }
    const leftDuration = splitAt - selectedClip.startSec;
    const rightDuration = clipEnd - splitAt;
    const right: TimelineClip = {
      ...selectedClip,
      id: selectedClip.id + "-b-" + Date.now().toString(36),
      startSec: splitAt,
      durationSec: rightDuration,
      fadeInSec: 0,
      fadeOutSec: selectedClip.fadeOutSec,
    };
    updateTimeline((current) => ({
      ...current,
      clips: current.clips.map((clip) =>
        clip.id === selectedClip.id ? { ...clip, durationSec: leftDuration, fadeOutSec: 0 } : clip,
      ).concat(right),
    }));
    setSelectedClipId(right.id);
    onMessage(`✅ Split ${selectedClip.label} at ${splitAt.toFixed(1)}s.`);
  }

  function trimStart() {
    if (!selectedClip) return;
    const newStart = clamp(selectedClip.startSec + EDIT_STEP, 0, selectedClip.startSec + selectedClip.durationSec - EDIT_STEP);
    const delta = newStart - selectedClip.startSec;
    updateClip(selectedClip.id, {
      startSec: newStart,
      durationSec: Math.max(EDIT_STEP, selectedClip.durationSec - delta),
    });
    onMessage(`Trimmed ${selectedClip.label} start by ${EDIT_STEP.toFixed(1)}s.`);
  }

  function trimEnd() {
    if (!selectedClip) return;
    updateClip(selectedClip.id, {
      durationSec: Math.max(EDIT_STEP, selectedClip.durationSec - EDIT_STEP),
    });
    onMessage(`Trimmed ${selectedClip.label} end by ${EDIT_STEP.toFixed(1)}s.`);
  }

  function extendEnd() {
    if (!selectedClip) return;
    const remaining = total - (selectedClip.startSec + selectedClip.durationSec);
    if (remaining <= 0.01) {
      onMessage("The clip already reaches the arrangement end.");
      return;
    }
    updateClip(selectedClip.id, {
      durationSec: selectedClip.durationSec + Math.min(EDIT_STEP, remaining),
    });
  }

  function moveClip(delta: number) {
    if (!selectedClip) return;
    updateClip(selectedClip.id, {
      startSec: clamp(selectedClip.startSec + delta, 0, Math.max(0, total - selectedClip.durationSec)),
    });
  }

  function duplicateSelected() {
    if (!selectedClip) return;
    const start = clamp(
      selectedClip.startSec + selectedClip.durationSec + EDIT_STEP,
      0,
      Math.max(0, total - selectedClip.durationSec),
    );
    const copy: TimelineClip = {
      ...selectedClip,
      id: selectedClip.id + "-copy-" + Date.now().toString(36),
      startSec: start,
    };
    updateTimeline((current) => ({ ...current, clips: [...current.clips, copy] }));
    setSelectedClipId(copy.id);
    onMessage(`✅ Duplicated ${selectedClip.label}.`);
  }

  function deleteSelected() {
    if (!selectedClip) return;
    const label = selectedClip.label;
    const id = selectedClip.id;
    updateTimeline((current) => ({
      ...current,
      clips: current.clips.filter((clip) => clip.id !== id),
    }));
    setSelectedClipId(null);
    onMessage(`Deleted ${label} clip.`);
  }

  function setFade(kind: "in" | "out") {
    if (!selectedClip || selectedClip.kind !== "audio") return;
    if (kind === "in") {
      updateClip(selectedClip.id, { fadeInSec: selectedClip.fadeInSec > 0 ? 0 : 0.5 });
    } else {
      updateClip(selectedClip.id, { fadeOutSec: selectedClip.fadeOutSec > 0 ? 0 : 0.5 });
    }
    onMessage(`${kind === "in" ? "Fade in" : "Fade out"} metadata ${kind === "in" ? "toggled" : "toggled"}.`);
  }

  function updateTrack(id: TrackId, patch: Partial<{ volume: number; muted: boolean; solo: boolean }>) {
    const previous = trackSettings[id] ?? defaultSetting(id);
    onTrackSettingsChange({
      ...trackSettings,
      [id]: { ...previous, ...patch },
    });
  }

  async function createParts() {
    setBuildingParts(true);
    onMessage("Generating Melody, Chords, Bass, Drums and Rhythm MIDI parts…");
    try {
      const bpm = config.bpm ?? 120;
      const beatsPerBarSeconds = (60 / bpm) * 4;
      const bars = Math.max(1, Math.min(64, Math.ceil(total / beatsPerBarSeconds)));
      const result = await generateMusicParts(jobId, {
        bpm,
        key: config.key ?? "C",
        scale: config.scale ?? "major",
        bars,
        seed: config.seed,
      });

      const downloaded: MusicPartFiles = { ...result.parts };
      for (const kind of Object.keys(result.parts) as Array<keyof typeof result.parts>) {
        downloaded[kind] = await downloadArtifact(jobId, kind, "mid");
      }

      onMusicPartsChange(downloaded);
      updateTimeline((current) => ({
        ...current,
        clips: [
          ...current.clips.filter((clip) => clip.kind !== "midi"),
          ...TRACKS.filter((track) => track.kind === "midi")
            .map((track) => downloaded[track.id as keyof MusicPartFiles] ? ({
              id: `clip-${track.id}-main-${Date.now().toString(36)}`,
              trackId: track.id,
              kind: "midi" as const,
              label: track.label,
              startSec: 0,
              durationSec: total,
              sourceUri: downloaded[track.id as keyof MusicPartFiles],
              sourcePart: track.id,
              fadeInSec: 0,
              fadeOutSec: 0,
            }) : null)
            .filter(Boolean) as TimelineClip[],
        ],
      }));
      onMessage("✅ Real MIDI parts created and placed on the timeline.");
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
      await Sharing.shareAsync(uri, { dialogTitle: "Share " + label + " MIDI", mimeType: "audio/midi" });
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Could not share MIDI.");
    }
  }

  return (
    <View style={{ backgroundColor: "#141821", borderRadius: 20, padding: 18, gap: 14, borderWidth: 1, borderColor: "#252b38" }}>
      <View style={{ gap: 5 }}>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 18, fontWeight: "800" }}>
          07 · Real Timeline + Clip Editor
        </Text>
        <Text selectable style={{ color: "#7f8899", fontSize: 12, lineHeight: 18 }}>
          Horizontal DAW-style timeline with section markers, playhead, zoom and non-destructive clip edits. Changes are saved locally with the project.
        </Text>
      </View>

      <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <Text selectable style={{ color: "#cbd3df", fontSize: 11, fontWeight: "800" }}>
          {total.toFixed(1)}s · Zoom {timeline.zoom.toFixed(0)} px/s
        </Text>
        <View style={{ flexDirection: "row", gap: 6 }}>
          <Button label="−" disabled={timeline.zoom <= MIN_ZOOM} onPress={() => updateTimeline((current) => ({ ...current, zoom: current.zoom - ZOOM_STEP }))} />
          <Button label="100%" onPress={() => updateTimeline((current) => ({ ...current, zoom: 48 }))} active={timeline.zoom === 48} />
          <Button label="+" disabled={timeline.zoom >= MAX_ZOOM} onPress={() => updateTimeline((current) => ({ ...current, zoom: current.zoom + ZOOM_STEP }))} />
        </View>
      </View>

      <View style={{ borderRadius: 14, overflow: "hidden", borderWidth: 1, borderColor: "#292f3c", backgroundColor: "#0d1015" }}>
        <View style={{ flexDirection: "row" }}>
          <View style={{ width: LABEL_W, height: RULER_H, justifyContent: "center", paddingHorizontal: 10, borderRightWidth: 1, borderRightColor: "#252b38" }}>
            <Text selectable style={{ color: "#697487", fontSize: 9, fontWeight: "900" }}>TRACK</Text>
          </View>
          <ScrollView
            ref={rulerRef}
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={{ width: timelineWidth }}
            onScroll={(event) => {
              const x = event.nativeEvent.contentOffset.x;
              setTimeline((current) => ({ ...current, playheadSec: current.playheadSec }));
              void x;
            }}
            scrollEventThrottle={16}
          >
            <Pressable
              onPress={(event) => {
                const x = clamp(event.nativeEvent.locationX, 0, timelineWidth);
                setPlayhead(snapTime(x / timeline.zoom));
              }}
              style={{ width: timelineWidth, height: RULER_H, justifyContent: "center" }}
            >
              {Array.from({ length: Math.ceil(total) + 1 }, (_, second) => (
                <View key={second} style={{ position: "absolute", left: second * timeline.zoom, top: 0, height: RULER_H, width: 1, backgroundColor: second % 5 === 0 ? "#394251" : "#252b33" }}>
                  <Text selectable style={{ position: "absolute", top: 3, left: 4, color: second % 5 === 0 ? "#a2acbb" : "#596273", fontSize: 9 }}>
                    {second}s
                  </Text>
                </View>
              ))}
              <View style={{ position: "absolute", left: timeline.playheadSec * timeline.zoom - 1, top: 0, bottom: 0, width: 2, backgroundColor: "#f4f6fa" }} />
            </Pressable>
          </ScrollView>
        </View>

        <View style={{ flexDirection: "row" }}>
          <View>
            {TRACKS.map((track) => (
              <View key={track.id} style={{ width: LABEL_W, height: TRACK_H, borderTopWidth: 1, borderTopColor: "#202633", justifyContent: "center", paddingHorizontal: 9, gap: 2 }}>
                <Text selectable numberOfLines={1} style={{ color: "#e0e5ed", fontSize: 11, fontWeight: "800" }}>{track.label}</Text>
                <Text selectable style={{ color: "#5e6878", fontSize: 8 }}>{track.kind.toUpperCase()}</Text>
              </View>
            ))}
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator contentContainerStyle={{ width: timelineWidth }}>
            <View style={{ width: timelineWidth }}>
              {TRACKS.map((track) => {
                const clips = clipForTrack(track.id);
                const setting = trackSettings[track.id] ?? defaultSetting(track.id);
                return (
                  <Pressable
                    key={track.id}
                    onPress={(event) => selectAt(snapTime(event.nativeEvent.locationX / timeline.zoom), track.id)}
                    style={{ width: timelineWidth, height: TRACK_H, borderTopWidth: 1, borderTopColor: "#202633" }}
                  >
                    {sectionMarkers.map((section) => (
                      <View key={section.name + section.index} pointerEvents="none" style={{ position: "absolute", left: section.start * timeline.zoom, top: 0, bottom: 0, width: Math.max(1, section.duration_seconds * timeline.zoom), backgroundColor: section.index % 2 === 0 ? "#11151c" : "#0e1218", borderRightWidth: 1, borderRightColor: "#202633" }}>
                        {section.index === 0 || section.duration_seconds * timeline.zoom > 48 ? (
                          <Text selectable numberOfLines={1} style={{ color: "#4f5969", fontSize: 8, padding: 5 }}>{section.name}</Text>
                        ) : null}
                      </View>
                    ))}
                    {clips.map((clip) => {
                      const selected = clip.id === selectedClipId;
                      return (
                        <Pressable
                          key={clip.id}
                          onPress={(event) => {
                            event.stopPropagation?.();
                            setSelectedClipId(clip.id);
                            setPlayhead(clamp(clip.startSec, 0, total));
                          }}
                          style={{
                            position: "absolute",
                            left: clip.startSec * timeline.zoom + 3,
                            top: 11,
                            width: Math.max(18, clip.durationSec * timeline.zoom - 6),
                            height: 40,
                            borderRadius: 7,
                            paddingHorizontal: 7,
                            justifyContent: "center",
                            backgroundColor: selected ? "#eef1f7" : clip.kind === "midi" ? "#263340" : "#2d333e",
                            borderWidth: 1,
                            borderColor: selected ? "#ffffff" : "#435064",
                            opacity: setting.muted ? 0.28 : setting.solo && soloExists ? 1 : 0.9,
                          }}
                        >
                          <Text selectable numberOfLines={1} style={{ color: selected ? "#0c1016" : "#cbd3df", fontSize: 9, fontWeight: "800" }}>
                            {clip.kind === "midi" ? "MIDI · " : ""}{clip.label}
                          </Text>
                          <Text selectable numberOfLines={1} style={{ color: selected ? "#384354" : "#718096", fontSize: 8 }}>
                            {clip.durationSec.toFixed(1)}s
                          </Text>
                          {clip.kind === "audio" && (clip.fadeInSec > 0 || clip.fadeOutSec > 0) ? (
                            <Text selectable style={{ position: "absolute", right: 4, bottom: 2, color: selected ? "#4f5969" : "#718096", fontSize: 7 }}>
                              {clip.fadeInSec > 0 ? "FI " : ""}{clip.fadeOutSec > 0 ? "FO" : ""}
                            </Text>
                          ) : null}
                        </Pressable>
                      );
                    })}
                    <View pointerEvents="none" style={{ position: "absolute", left: timeline.playheadSec * timeline.zoom - 1, top: 0, bottom: 0, width: 2, backgroundColor: "#f4f6fa" }} />
                  </Pressable>
                );
              })}
            </View>
          </ScrollView>
        </View>
      </View>

      {selectedClip ? (
        <View style={{ backgroundColor: "#0e1117", borderRadius: 14, borderWidth: 1, borderColor: "#2b3340", padding: 12, gap: 9 }}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", gap: 8 }}>
            <View style={{ flex: 1, gap: 2 }}>
              <Text selectable style={{ color: "#f2f5f8", fontWeight: "800" }}>{selectedClip.label}</Text>
              <Text selectable style={{ color: "#707a8a", fontSize: 10 }}>
                {selectedClip.kind.toUpperCase()} · {selectedClip.startSec.toFixed(1)}s → {(selectedClip.startSec + selectedClip.durationSec).toFixed(1)}s
              </Text>
            </View>
            <Button label="Clear" onPress={() => setSelectedClipId(null)} />
          </View>

          <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
            <Button label="Split @ Playhead" onPress={splitSelected} />
            <Button label="◀ Move" onPress={() => moveClip(-EDIT_STEP)} />
            <Button label="Move ▶" onPress={() => moveClip(EDIT_STEP)} />
            <Button label="Trim Start" onPress={trimStart} />
            <Button label="Trim End" onPress={trimEnd} />
            <Button label="Extend End" onPress={extendEnd} />
            <Button label="Duplicate" onPress={duplicateSelected} />
            {selectedClip.kind === "audio" ? (
              <>
                <Button label="Fade In" active={selectedClip.fadeInSec > 0} onPress={() => setFade("in")} />
                <Button label="Fade Out" active={selectedClip.fadeOutSec > 0} onPress={() => setFade("out")} />
              </>
            ) : null}
            <Button label="Delete" onPress={deleteSelected} />
          </View>

          <View style={{ borderTopWidth: 1, borderTopColor: "#202633", paddingTop: 8, gap: 4 }}>
            <Text selectable style={{ color: "#8e98a8", fontSize: 10, fontWeight: "800" }}>AI CLIP EDITOR</Text>
            <Text selectable style={{ color: "#5e6878", fontSize: 10, lineHeight: 15 }}>
              Select a clip here first. Clip-specific Clean / Pitch / Timing / Harmony / Regenerate actions will be connected to dedicated server jobs; they are not faked as local DSP.
            </Text>
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
              {["Clean", "Fix Pitch", "Fix Timing", "Regenerate", "Harmony", "Extend"].map((label) => (
                <Button
                  key={label}
                  label={label}
                  disabled
                  onPress={() => undefined}
                />
              ))}
            </View>
          </View>
        </View>
      ) : (
        <Text selectable style={{ color: "#667183", fontSize: 11, lineHeight: 17 }}>
          Tap a clip to select it. Tap the ruler or a track area to move the playhead. Then use split, trim, move, duplicate, fade and delete.
        </Text>
      )}

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#8f98aa", fontSize: 10, fontWeight: "900" }}>TRACK CONTROLS</Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
          {TRACKS.map((track) => {
            const current = trackSettings[track.id] ?? defaultSetting(track.id);
            return (
              <View key={track.id} style={{ width: "100%", borderRadius: 12, backgroundColor: "#0e1117", borderWidth: 1, borderColor: current.solo ? "#626d81" : "#252b38", padding: 9, gap: 7 }}>
                <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                  <View style={{ flex: 1 }}>
                    <Text selectable style={{ color: "#dce2ea", fontSize: 11, fontWeight: "800" }}>{track.label}</Text>
                    <Text selectable style={{ color: "#5e6878", fontSize: 8 }}>{track.kind === "midi" ? "MIDI lane" : "Audio lane"} · {Math.round(current.volume * 100)}%</Text>
                  </View>
                  <Button label="M" active={current.muted} onPress={() => updateTrack(track.id, { muted: !current.muted })} />
                  <Button label="S" active={current.solo} onPress={() => updateTrack(track.id, { solo: !current.solo })} />
                  <Button label="VOL −" disabled={current.volume <= 0} onPress={() => updateTrack(track.id, { volume: Number(Math.max(0, current.volume - 0.1).toFixed(2)) })} />
                  <Button label="VOL +" disabled={current.volume >= 1} onPress={() => updateTrack(track.id, { volume: Number(Math.min(1, current.volume + 0.1).toFixed(2)) })} />
                  {musicParts?.[track.id as keyof MusicPartFiles] && track.kind === "midi" ? (
                    <Button label="Share MIDI" onPress={() => void shareMidi(musicParts[track.id as keyof MusicPartFiles]!, track.label)} />
                  ) : null}
                </View>
              </View>
            );
          })}
        </View>
      </View>

      <Button
        label={buildingParts ? "Generating MIDI…" : musicParts?.arrangement ? "Regenerate MIDI Parts" : "Generate MIDI Parts"}
        disabled={buildingParts}
        onPress={() => void createParts()}
      />
    </View>
  );
}
