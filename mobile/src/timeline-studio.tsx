import { useEffect, useMemo, useRef, useState } from "react";
import * as Sharing from "expo-sharing";
import { useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import { PanResponder, Pressable, ScrollView, Text, View } from "react-native";

import {
  artifactUrl,
  downloadArtifact,
  downloadClipEdit,
  downloadMidiEdit,
  editSongClip,
  saveMidiEdits,
  generateMusicParts,
  getMidiNotes,
  renderTimeline,
} from "./api";
import type { ClipEditAction } from "./api";
import type {
  MidiNote,
  MusicPartFiles,
  SongConfig,
  TimelineClip,
  TimelineState,
  TrackMixSettings,
  TrackMixState,
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

function defaultSetting(id: TrackId): TrackMixState {
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

function snapTime(seconds: number) {
  return Math.round(seconds / EDIT_STEP) * EDIT_STEP;
}

function totalSeconds(config: SongConfig) {
  return Math.max(
    1,
    config.sections.reduce((sum, section) => sum + Math.max(0, section.duration_seconds), 0),
  );
}

function normalizeTimeline(timeline: TimelineState): TimelineState {
  return {
    ...timeline,
    playheadSec: Number.isFinite(timeline.playheadSec) ? timeline.playheadSec : 0,
    zoom: Number.isFinite(timeline.zoom) ? clamp(timeline.zoom, MIN_ZOOM, MAX_ZOOM) : 48,
    clips: timeline.clips.map((clip) => ({
      ...clip,
      sourceOffsetSec: Number.isFinite(clip.sourceOffsetSec) ? clip.sourceOffsetSec : 0,
      fadeInSec: Number.isFinite(clip.fadeInSec) ? clip.fadeInSec : 0,
      fadeOutSec: Number.isFinite(clip.fadeOutSec) ? clip.fadeOutSec : 0,
    })),
  };
}

function defaultTimeline(
  config: SongConfig,
  artifacts: { vocal?: string; backing?: string },
  musicParts?: MusicPartFiles,
): TimelineState {
  const total = totalSeconds(config);
  const clips: TimelineClip[] = [];

  if (artifacts.vocal) {
    clips.push({
      id: "clip-vocal-main",
      trackId: "vocal",
      kind: "audio",
      label: "Vocal",
      startSec: 0,
      durationSec: total,
      sourceOffsetSec: 0,
      sourceUri: artifacts.vocal,
      sourceArtifact: "vocal",
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }

  if (artifacts.backing) {
    clips.push({
      id: "clip-backing-main",
      trackId: "backing",
      kind: "audio",
      label: "AI Backing",
      startSec: 0,
      durationSec: total,
      sourceOffsetSec: 0,
      sourceUri: artifacts.backing,
      sourceArtifact: "backing",
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }

  for (const track of TRACKS.filter((item) => item.kind === "midi")) {
    const uri = musicParts?.[track.id as keyof MusicPartFiles];
    if (!uri) continue;
    const sourcePart = track.id as NonNullable<TimelineClip["sourcePart"]>;
    clips.push({
      id: `clip-${track.id}-main`,
      trackId: track.id,
      kind: "midi",
      label: track.label,
      startSec: 0,
      durationSec: total,
      sourceUri: uri,
      sourcePart,
      fadeInSec: 0,
      fadeOutSec: 0,
    });
  }

  return { clips, playheadSec: 0, zoom: 48 };
}

function ClipBlock({
  clip,
  selected,
  current,
  soloExists,
  zoom,
  total,
  onSelect,
  onMove,
  onResize,
}: {
  clip: TimelineClip;
  selected: boolean;
  current: TrackMixState;
  soloExists: boolean;
  zoom: number;
  total: number;
  onSelect: (clipId: string) => void;
  onMove: (clipId: string, baseStart: number, deltaSeconds: number) => void;
  onResize: (
    clipId: string,
    baseStart: number,
    baseDuration: number,
    edge: "left" | "right",
    deltaSeconds: number,
  ) => void;
}) {
  const dragStart = useRef({ start: clip.startSec, duration: clip.durationSec });

  const moveResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => false,
        onMoveShouldSetPanResponder: (_, gesture) =>
          Math.abs(gesture.dx) > Math.abs(gesture.dy) && Math.abs(gesture.dx) > 5,
        onPanResponderGrant: () => {
          dragStart.current = { start: clip.startSec, duration: clip.durationSec };
          onSelect(clip.id);
        },
        onPanResponderMove: (_, gesture) => {
          onMove(clip.id, dragStart.current.start, gesture.dx / zoom);
        },
        onPanResponderTerminationRequest: () => false,
      }),
    [clip.id, clip.startSec, clip.durationSec, onMove, onSelect, zoom],
  );

  const leftResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          dragStart.current = { start: clip.startSec, duration: clip.durationSec };
          onSelect(clip.id);
        },
        onPanResponderMove: (_, gesture) => {
          onResize(
            clip.id,
            dragStart.current.start,
            dragStart.current.duration,
            "left",
            gesture.dx / zoom,
          );
        },
        onPanResponderTerminationRequest: () => false,
      }),
    [clip.id, clip.startSec, clip.durationSec, onResize, onSelect, zoom],
  );

  const rightResponder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onPanResponderGrant: () => {
          dragStart.current = { start: clip.startSec, duration: clip.durationSec };
          onSelect(clip.id);
        },
        onPanResponderMove: (_, gesture) => {
          onResize(
            clip.id,
            dragStart.current.start,
            dragStart.current.duration,
            "right",
            gesture.dx / zoom,
          );
        },
        onPanResponderTerminationRequest: () => false,
      }),
    [clip.id, clip.startSec, clip.durationSec, onResize, onSelect, zoom],
  );

  return (
    <View
      {...moveResponder.panHandlers}
      style={{
        position: "absolute",
        left: clip.startSec * zoom + 3,
        top: 11,
        width: Math.max(22, clip.durationSec * zoom - 6),
        height: 40,
        borderRadius: 7,
        backgroundColor: selected ? "#eef1f7" : clip.kind === "midi" ? "#263340" : "#2d333e",
        borderWidth: 1,
        borderColor: selected ? "#ffffff" : "#435064",
        opacity: current.muted ? 0.28 : current.solo && soloExists ? 1 : 0.9,
        overflow: "hidden",
      }}
    >
      <Pressable
        onPress={() => onSelect(clip.id)}
        style={{ flex: 1, justifyContent: "center", paddingHorizontal: 10 }}
      >
        <Text
          selectable
          numberOfLines={1}
          style={{
            color: selected ? "#0c1016" : "#cbd3df",
            fontSize: 9,
            fontWeight: "800",
            paddingRight: 16,
          }}
        >
          {clip.kind === "midi" ? "MIDI · " : ""}
          {clip.label}
        </Text>
        <Text
          selectable
          numberOfLines={1}
          style={{ color: selected ? "#384354" : "#718096", fontSize: 8 }}
        >
          {clip.durationSec.toFixed(1)}s
        </Text>
      </Pressable>

      <View
        {...leftResponder.panHandlers}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: 13,
          backgroundColor: selected ? "#cbd3df" : "#455064",
          opacity: 0.8,
        }}
      />
      <View
        {...rightResponder.panHandlers}
        style={{
          position: "absolute",
          right: 0,
          top: 0,
          bottom: 0,
          width: 13,
          backgroundColor: selected ? "#cbd3df" : "#455064",
          opacity: 0.8,
        }}
      />

      {clip.kind === "audio" && (clip.fadeInSec > 0 || clip.fadeOutSec > 0) ? (
        <Text
          selectable
          style={{
            position: "absolute",
            right: 16,
            bottom: 2,
            color: selected ? "#4f5969" : "#718096",
            fontSize: 7,
          }}
        >
          {clip.fadeInSec > 0 ? "FI " : ""}
          {clip.fadeOutSec > 0 ? "FO" : ""}
        </Text>
      ) : null}
    </View>
  );
}

function noteName(midi: number) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return `${names[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

function PianoRoll({
  jobId,
  clip,
  bpm,
  onSaved,
}: {
  jobId: string;
  clip: TimelineClip;
  bpm: number;
  onSaved: (uri: string, part: NonNullable<TimelineClip["sourcePart"]>) => void;
}) {
  const [notes, setNotes] = useState<MidiNote[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNote, setSelectedNote] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadNotes() {
      if (!clip.sourcePart) {
        setNotes([]);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const result = await getMidiNotes(jobId, clip.sourcePart);
        if (active) setNotes(result.notes);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : "Could not load MIDI notes.");
      } finally {
        if (active) setLoading(false);
      }
    }

    void loadNotes();
    return () => {
      active = false;
    };
  }, [clip.id, clip.sourcePart, jobId]);

  function updateSelectedNote(patch: Partial<MidiNote>) {
    if (selectedNote === null) return;
    setNotes((current) =>
      current.map((note, index) => (index === selectedNote ? { ...note, ...patch } : note)),
    );
  }

  function deleteSelectedNote() {
    if (selectedNote === null) return;
    setNotes((current) => current.filter((_, index) => index !== selectedNote));
    setSelectedNote(null);
  }

  function duplicateSelectedNote() {
    if (selectedNote === null) return;
    setNotes((current) => {
      const source = current[selectedNote];
      if (!source) return current;
      const copy = { ...source, startBeat: source.startBeat + Math.max(0.25, source.durationBeat) };
      return [...current, copy].sort((a, b) => a.startBeat - b.startBeat || a.note - b.note);
    });
  }

  async function persistNotes() {
    if (!clip.sourcePart || saving) return;
    setSaving(true);
    try {
      const result = await saveMidiEdits(jobId, {
        part: clip.sourcePart,
        notes: notes.map((note) => ({
          note: Math.round(clamp(note.note, 0, 127)),
          velocity: Math.round(clamp(note.velocity, 1, 127)),
          startBeat: Math.max(0, note.startBeat),
          durationBeat: Math.max(0.01, note.durationBeat),
        })),
        bpm,
      });
      const uri = await downloadMidiEdit(jobId, clip.sourcePart);
      onSaved(uri, clip.sourcePart);
      setSelectedNote(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save MIDI edits.");
    } finally {
      setSaving(false);
    }
  }

  const pitchBounds = useMemo(() => {
    if (!notes.length) return { min: 48, max: 72 };
    const min = Math.min(...notes.map((item) => item.note));
    const max = Math.max(...notes.map((item) => item.note));
    return { min: Math.max(24, min - 2), max: Math.min(108, max + 2) };
  }, [notes]);

  const maxBeat = notes.length
    ? Math.max(...notes.map((item) => item.startBeat + item.durationBeat), 4)
    : 4;
  const rowHeight = 18;
  const beatWidth = 34;
  const rowCount = Math.max(1, pitchBounds.max - pitchBounds.min + 1);
  const width = Math.max(520, Math.ceil(maxBeat * beatWidth + 40));
  const height = rowCount * rowHeight;

  return (
    <View
      style={{
        borderRadius: 14,
        borderWidth: 1,
        borderColor: "#2b3340",
        backgroundColor: "#0b0e13",
        overflow: "hidden",
      }}
    >
      <View style={{ padding: 12, gap: 3 }}>
        <Text selectable style={{ color: "#f2f5f8", fontWeight: "800", fontSize: 13 }}>
          MIDI Piano Roll · {clip.label}
        </Text>
        <Text selectable style={{ color: "#667183", fontSize: 10 }}>
          {bpm} BPM · {notes.length} notes · drag/trim edits on the parent MIDI clip
        </Text>
      </View>

      {loading ? (
        <Text selectable style={{ color: "#8893a3", fontSize: 11, padding: 12 }}>
          Loading MIDI note data…
        </Text>
      ) : error ? (
        <Text selectable style={{ color: "#ff9a9a", fontSize: 11, lineHeight: 17, padding: 12 }}>
          {error}
        </Text>
      ) : !notes.length ? (
        <Text selectable style={{ color: "#70798a", fontSize: 11, lineHeight: 17, padding: 12 }}>
          No parsed notes yet. Generate the MIDI part first.
        </Text>
      ) : (
        <View style={{ flexDirection: "row", height: Math.min(420, height) }}>
          <View
            style={{
              width: 52,
              backgroundColor: "#11151c",
              borderRightWidth: 1,
              borderRightColor: "#252b38",
            }}
          >
            {Array.from({ length: rowCount }, (_, index) => {
              const pitch = pitchBounds.max - index;
              return (
                <View
                  key={pitch}
                  style={{
                    height: rowHeight,
                    borderBottomWidth: 1,
                    borderBottomColor: "#1c222d",
                    justifyContent: "center",
                    paddingHorizontal: 5,
                  }}
                >
                  <Text selectable style={{ color: "#687385", fontSize: 8 }}>
                    {noteName(pitch)}
                  </Text>
                </View>
              );
            })}
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator>
            <View style={{ width, height }}>
              {Array.from({ length: Math.ceil(maxBeat) + 1 }, (_, beat) => (
                <View
                  key={beat}
                  pointerEvents="none"
                  style={{
                    position: "absolute",
                    left: beat * beatWidth,
                    top: 0,
                    bottom: 0,
                    width: 1,
                    backgroundColor: beat % 4 === 0 ? "#394251" : "#202633",
                  }}
                />
              ))}

              {notes.map((note, index) => {
                const top = (pitchBounds.max - note.note) * rowHeight;
                return (
                  <Pressable
                    key={`${note.note}-${note.startBeat}-${index}`}
                    onPress={() => setSelectedNote(index)}
                    style={{
                      position: "absolute",
                      left: note.startBeat * beatWidth + 1,
                      top,
                      width: Math.max(3, note.durationBeat * beatWidth - 2),
                      height: Math.max(10, rowHeight - 3),
                      borderRadius: 3,
                      backgroundColor: "#8792a4",
                      borderWidth: 1,
                      borderColor: selectedNote === index ? "#ffffff" : "#b9c1ce",
                    }}
                  />
                );
              })}

              {Array.from({ length: Math.ceil(maxBeat / 4) }, (_, index) => (
                <Text
                  key={index}
                  selectable
                  style={{
                    position: "absolute",
                    left: index * 4 * beatWidth + 4,
                    top: 3,
                    color: "#616c7c",
                    fontSize: 8,
                  }}
                >
                  Bar {index + 1}
                </Text>
              ))}
            </View>
          </ScrollView>
        </View>
      )}

      {!loading && !error && notes.length ? (
        <View style={{ borderTopWidth: 1, borderTopColor: "#202633", padding: 12, gap: 8 }}>
          <Text selectable style={{ color: "#7f899a", fontSize: 9, fontWeight: "900" }}>
            NOTE EDITOR
          </Text>
          {selectedNote === null ? (
            <Text selectable style={{ color: "#5f6979", fontSize: 10 }}>
              Tap a note, then adjust pitch/duration or delete it.
            </Text>
          ) : (
            <>
              <Text selectable style={{ color: "#dce2ea", fontSize: 11, fontWeight: "800" }}>
                {noteName(notes[selectedNote]?.note ?? 60)} · beat {(notes[selectedNote]?.startBeat ?? 0).toFixed(2)}
              </Text>
              <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
                <Button label="Pitch −" onPress={() => updateSelectedNote({ note: (notes[selectedNote]?.note ?? 60) - 1 })} />
                <Button label="Pitch +" onPress={() => updateSelectedNote({ note: (notes[selectedNote]?.note ?? 60) + 1 })} />
                <Button label="Length −" onPress={() => updateSelectedNote({ durationBeat: Math.max(0.05, (notes[selectedNote]?.durationBeat ?? 0.25) - 0.25) })} />
                <Button label="Length +" onPress={() => updateSelectedNote({ durationBeat: (notes[selectedNote]?.durationBeat ?? 0.25) + 0.25 })} />
                <Button label="Duplicate Note" onPress={duplicateSelectedNote} />
                <Button label="Delete Note" onPress={deleteSelectedNote} />
                <Button label={saving ? "Saving…" : "Save MIDI"} disabled={saving} active={!saving} onPress={() => void persistNotes()} />
              </View>
            </>
          )}
        </View>
      ) : null}
    </View>
  );
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
  onTimelineRendered,
  onMessage,
}: {
  jobId: string;
  config: SongConfig;
  artifacts: { final?: string; vocal?: string; backing?: string };
  musicParts?: MusicPartFiles;
  trackSettings: TrackMixSettings;
  initialTimeline?: TimelineState;
  onTimelineChange: (timeline: TimelineState) => void;
  onTrackSettingsChange: (settings: TrackMixSettings) => void;
  onMusicPartsChange: (parts: MusicPartFiles) => void;
  onTimelineRendered: (uri: string) => void;
  onMessage: (message: string) => void;
}) {
  const [timeline, setTimeline] = useState<TimelineState>(
    () => normalizeTimeline(initialTimeline ?? defaultTimeline(config, artifacts, musicParts)),
  );
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [buildingParts, setBuildingParts] = useState(false);
  const [editingAction, setEditingAction] = useState<ClipEditAction | null>(null);
  const [renderingTimeline, setRenderingTimeline] = useState(false);
  const lastPersistedPlayback = useRef(-1);

  let playbackSource: string | null = artifacts.final ?? null;
  if (!playbackSource) {
    try {
      playbackSource = artifactUrl(jobId, "final");
    } catch {
      playbackSource = null;
    }
  }

  const player = useAudioPlayer(playbackSource, { updateInterval: 100 });
  const playback = useAudioPlayerStatus(player);

  useEffect(() => {
    if (!playback.playing) {
      const seconds = clamp(playback.currentTime || 0, 0, totalSeconds(config));
      if (Math.abs(seconds - lastPersistedPlayback.current) >= 0.1) {
        lastPersistedPlayback.current = seconds;
        setTimeline((current) => ({ ...current, playheadSec: snapTime(seconds) }));
      }
    }
  }, [playback.currentTime, playback.playing, config]);

  useEffect(() => {
    if (initialTimeline) setTimeline(normalizeTimeline(initialTimeline));
  }, [initialTimeline]);

  useEffect(() => {
    onTimelineChange(timeline);
  }, [timeline, onTimelineChange]);

  const total = totalSeconds(config);
  const displayPlayhead = playback.playing
    ? clamp(playback.currentTime || 0, 0, total)
    : timeline.playheadSec;
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

  function seekAndSetPlayhead(seconds: number) {
    const target = clamp(snapTime(seconds), 0, total);
    updateTimeline((current) => ({ ...current, playheadSec: target }));
    void player.seekTo(target);
  }

  function selectAt(seconds: number, trackId: TrackId) {
    const candidate = timeline.clips.find(
      (clip) =>
        clip.trackId === trackId &&
        seconds >= clip.startSec &&
        seconds <= clip.startSec + clip.durationSec,
    );
    setSelectedClipId(candidate?.id ?? null);
    seekAndSetPlayhead(seconds);
  }

  function selectClip(id: string) {
    const clip = timeline.clips.find((item) => item.id === id);
    setSelectedClipId(id);
    if (clip) seekAndSetPlayhead(clip.startSec);
  }

  function updateClip(id: string, patch: Partial<TimelineClip>) {
    updateTimeline((current) => ({
      ...current,
      clips: current.clips.map((clip) => (clip.id === id ? { ...clip, ...patch } : clip)),
    }));
  }

  function moveClipDrag(id: string, baseStart: number, deltaSeconds: number) {
    const clip = timeline.clips.find((item) => item.id === id);
    if (!clip) return;
    const nextStart = clamp(
      snapTime(baseStart + deltaSeconds),
      0,
      Math.max(0, total - clip.durationSec),
    );
    updateClip(id, { startSec: nextStart });
  }

  function resizeClipDrag(
    id: string,
    baseStart: number,
    baseDuration: number,
    edge: "left" | "right",
    deltaSeconds: number,
  ) {
    if (edge === "left") {
      const nextStart = clamp(
        snapTime(baseStart + deltaSeconds),
        0,
        baseStart + baseDuration - EDIT_STEP,
      );
      const sourceShift = nextStart - baseStart;
      const clip = timeline.clips.find((item) => item.id === id);
      updateClip(id, {
        startSec: nextStart,
        durationSec: Math.max(EDIT_STEP, baseDuration - sourceShift),
        ...(clip ? { sourceOffsetSec: clip.sourceOffsetSec + sourceShift } : {}),
      });
      return;
    }

    updateClip(id, {
      durationSec: clamp(
        snapTime(baseDuration + deltaSeconds),
        EDIT_STEP,
        Math.max(EDIT_STEP, total - baseStart),
      ),
    });
  }

  function splitSelected() {
    if (!selectedClip) return;
    const splitAt = clamp(snapTime(timeline.playheadSec), 0, total);
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
      sourceOffsetSec: selectedClip.sourceOffsetSec + leftDuration,
      fadeInSec: 0,
      fadeOutSec: selectedClip.fadeOutSec,
    };

    updateTimeline((current) => ({
      ...current,
      clips: current.clips
        .map((clip) =>
          clip.id === selectedClip.id
            ? { ...clip, durationSec: leftDuration, fadeOutSec: 0 }
            : clip,
        )
        .concat(right),
    }));
    setSelectedClipId(right.id);
    onMessage(`✅ Split ${selectedClip.label} at ${splitAt.toFixed(1)}s.`);
  }

  function trimStart() {
    if (!selectedClip) return;
    const maxStart = selectedClip.startSec + selectedClip.durationSec - EDIT_STEP;
    const newStart = clamp(selectedClip.startSec + EDIT_STEP, 0, maxStart);
    const delta = newStart - selectedClip.startSec;
    updateClip(selectedClip.id, {
      startSec: newStart,
      durationSec: Math.max(EDIT_STEP, selectedClip.durationSec - delta),
      sourceOffsetSec: selectedClip.sourceOffsetSec + delta,
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
      startSec: clamp(
        snapTime(selectedClip.startSec + delta),
        0,
        Math.max(0, total - selectedClip.durationSec),
      ),
    });
  }

  function duplicateSelected() {
    if (!selectedClip) return;
    const start = clamp(
      snapTime(selectedClip.startSec + selectedClip.durationSec + EDIT_STEP),
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
    onMessage(`${kind === "in" ? "Fade in" : "Fade out"} metadata toggled.`);
  }

  async function runClipAIAction(action: ClipEditAction) {
    if (!selectedClip || selectedClip.kind !== "audio") {
      onMessage("Select an audio clip before running an AI clip action.");
      return;
    }

    const source = selectedClip.sourceArtifact;
    if (!source) {
      onMessage("This clip has no editable audio source.");
      return;
    }
    if ((action === "fix-pitch" || action === "fix-timing") && source !== "vocal") {
      onMessage("Pitch and timing correction are currently available for vocal clips only.");
      return;
    }

    setEditingAction(action);
    onMessage(`Running ${action} on ${selectedClip.label}…`);

    try {
      const result = await editSongClip(jobId, {
        source,
        action,
        strength: 0.65,
      });
      const localUri = await downloadClipEdit(jobId, result.edit_id, "wav");
      updateClip(selectedClip.id, {
        sourceUri: localUri,
        label: `${selectedClip.label} · ${action}`,
      });
      onMessage(`✅ ${action} finished. The selected clip now points to the new local WAV.`);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "AI clip edit failed.");
    } finally {
      setEditingAction(null);
    }
  }

  function updateTrack(id: TrackId, patch: Partial<TrackMixState>) {
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
      for (const kind of Object.keys(result.parts) as Array<keyof MusicPartFiles>) {
        downloaded[kind] = await downloadArtifact(jobId, kind, "mid");
      }

      onMusicPartsChange(downloaded);
      updateTimeline((current) => ({
        ...current,
        clips: [
          ...current.clips.filter((clip) => clip.kind !== "midi"),
          ...TRACKS.filter((track) => track.kind === "midi")
            .map((track) => {
              const sourceUri = downloaded[track.id as keyof MusicPartFiles];
              if (!sourceUri) return null;
              const sourcePart = track.id as NonNullable<TimelineClip["sourcePart"]>;
              return {
                id: `clip-${track.id}-main-${Date.now().toString(36)}`,
                trackId: track.id,
                kind: "midi" as const,
                label: track.label,
                startSec: 0,
                durationSec: total,
                sourceOffsetSec: 0,
                sourceUri,
                sourcePart,
                fadeInSec: 0,
                fadeOutSec: 0,
              };
            })
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

  async function renderCurrentTimeline() {
    if (renderingTimeline) return;
    if (!timeline.clips.some((clip) => clip.kind === "audio" && clip.durationSec > 0)) {
      onMessage("Add at least one audio clip before rendering the timeline.");
      return;
    }

    setRenderingTimeline(true);
    onMessage("Rendering edited timeline to WAV…");
    try {
      await renderTimeline(jobId, {
        timeline,
        trackMix: trackSettings,
        bpm: config.bpm ?? 120,
        targetPeak: 0.95,
        compressionRatio: 2,
        saturation: 0.08,
      });
      const uri = await downloadArtifact(jobId, "timeline", "wav");
      onTimelineRendered(uri);
      onMessage("✅ Timeline rendered to a new WAV and saved locally.");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Timeline render failed.");
    } finally {
      setRenderingTimeline(false);
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
    <View style={{ backgroundColor: "#141821", borderRadius: 20, padding: 18, gap: 14, borderWidth: 1, borderColor: "#252b38" }}>
      <View style={{ gap: 5 }}>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 18, fontWeight: "800" }}>
          07 · Real Timeline + Clip Editor
        </Text>
        <Text selectable style={{ color: "#7f8899", fontSize: 12, lineHeight: 18 }}>
          Drag clips to move them. Use the left/right handles to trim. Playback playhead follows the master audio clock, and MIDI clips open a piano-roll foundation.
        </Text>
      </View>

      <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <View style={{ gap: 2 }}>
          <Text selectable style={{ color: "#cbd3df", fontSize: 11, fontWeight: "800" }}>
            {total.toFixed(1)}s · Zoom {timeline.zoom.toFixed(0)} px/s
          </Text>
          <Text selectable style={{ color: playback.playing ? "#dce2ea" : "#697487", fontSize: 9 }}>
            {displayPlayhead.toFixed(1)}s · {playback.playing ? "PLAYING" : "PAUSED"}
          </Text>
        </View>
        <View style={{ flexDirection: "row", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <Button
            label={playback.playing ? "Pause" : "Play"}
            disabled={!playbackSource}
            active={playback.playing}
            onPress={() => {
              if (!playbackSource) return;
              if (playback.playing) {
                player.pause();
              } else {
                void player.seekTo(timeline.playheadSec).then(() => player.play());
              }
            }}
          />
          <Button
            label="↖ Seek"
            disabled={!playbackSource}
            onPress={() => void player.seekTo(timeline.playheadSec)}
          />
          <Button
            label="−"
            disabled={timeline.zoom <= MIN_ZOOM}
            onPress={() => updateTimeline((current) => ({ ...current, zoom: current.zoom - ZOOM_STEP }))}
          />
          <Button
            label="100%"
            onPress={() => updateTimeline((current) => ({ ...current, zoom: 48 }))}
            active={timeline.zoom === 48}
          />
          <Button
            label="+"
            disabled={timeline.zoom >= MAX_ZOOM}
            onPress={() => updateTimeline((current) => ({ ...current, zoom: current.zoom + ZOOM_STEP }))}
          />
        </View>
      </View>

      {!artifacts.final ? (
        <Text selectable style={{ color: "#778294", fontSize: 10, lineHeight: 16 }}>
          Save Final WAV above to enable timeline playback. Editing, clip layout and MIDI tools remain available without it.
        </Text>
      ) : null}

      <View style={{ borderRadius: 14, overflow: "hidden", borderWidth: 1, borderColor: "#292f3c", backgroundColor: "#0d1015" }}>
        <View style={{ flexDirection: "row" }}>
          <View style={{ width: LABEL_W, height: RULER_H, justifyContent: "center", paddingHorizontal: 10, borderRightWidth: 1, borderRightColor: "#252b38" }}>
            <Text selectable style={{ color: "#697487", fontSize: 9, fontWeight: "900" }}>TRACK</Text>
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <Pressable
              onPress={(event) => {
                const x = clamp(event.nativeEvent.locationX, 0, timelineWidth);
                seekAndSetPlayhead(x / timeline.zoom);
              }}
              style={{ width: timelineWidth, height: RULER_H, justifyContent: "center" }}
            >
              {Array.from({ length: Math.ceil(total) + 1 }, (_, second) => (
                <View
                  key={second}
                  style={{
                    position: "absolute",
                    left: second * timeline.zoom,
                    top: 0,
                    height: RULER_H,
                    width: 1,
                    backgroundColor: second % 5 === 0 ? "#394251" : "#252b33",
                  }}
                >
                  {second % 5 === 0 ? (
                    <Text selectable style={{ position: "absolute", top: 3, left: 4, color: "#a2acbb", fontSize: 9 }}>
                      {second}s
                    </Text>
                  ) : null}
                </View>
              ))}

              <View
                pointerEvents="none"
                style={{
                  position: "absolute",
                  left: displayPlayhead * timeline.zoom - 1,
                  top: 0,
                  bottom: 0,
                  width: 2,
                  backgroundColor: "#f4f6fa",
                }}
              />
            </Pressable>
          </ScrollView>
        </View>

        <View style={{ flexDirection: "row" }}>
          <View>
            {TRACKS.map((track) => (
              <View
                key={track.id}
                style={{
                  width: LABEL_W,
                  height: TRACK_H,
                  borderTopWidth: 1,
                  borderTopColor: "#202633",
                  justifyContent: "center",
                  paddingHorizontal: 9,
                  gap: 2,
                }}
              >
                <Text selectable numberOfLines={1} style={{ color: "#e0e5ed", fontSize: 11, fontWeight: "800" }}>
                  {track.label}
                </Text>
                <Text selectable style={{ color: "#5e6878", fontSize: 8 }}>{track.kind.toUpperCase()}</Text>
              </View>
            ))}
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator contentContainerStyle={{ width: timelineWidth }}>
            <View style={{ width: timelineWidth }}>
              {TRACKS.map((track) => {
                const clips = timeline.clips.filter((clip) => clip.trackId === track.id);
                const currentSetting = trackSettings[track.id] ?? defaultSetting(track.id);

                return (
                  <Pressable
                    key={track.id}
                    onPress={(event) =>
                      selectAt(snapTime(event.nativeEvent.locationX / timeline.zoom), track.id)
                    }
                    style={{
                      width: timelineWidth,
                      height: TRACK_H,
                      borderTopWidth: 1,
                      borderTopColor: "#202633",
                    }}
                  >
                    {sectionMarkers.map((section) => (
                      <View
                        key={section.name + section.index}
                        pointerEvents="none"
                        style={{
                          position: "absolute",
                          left: section.start * timeline.zoom,
                          top: 0,
                          bottom: 0,
                          width: Math.max(1, section.duration_seconds * timeline.zoom),
                          backgroundColor: section.index % 2 === 0 ? "#11151c" : "#0e1218",
                          borderRightWidth: 1,
                          borderRightColor: "#202633",
                        }}
                      >
                        {section.index === 0 || section.duration_seconds * timeline.zoom > 48 ? (
                          <Text selectable numberOfLines={1} style={{ color: "#4f5969", fontSize: 8, padding: 5 }}>
                            {section.name}
                          </Text>
                        ) : null}
                      </View>
                    ))}

                    {clips.map((clip) => (
                      <ClipBlock
                        key={clip.id}
                        clip={clip}
                        selected={clip.id === selectedClipId}
                        current={currentSetting}
                        soloExists={soloExists}
                        zoom={timeline.zoom}
                        total={total}
                        onSelect={selectClip}
                        onMove={moveClipDrag}
                        onResize={resizeClipDrag}
                      />
                    ))}

                    <View
                      pointerEvents="none"
                      style={{
                        position: "absolute",
                        left: displayPlayhead * timeline.zoom - 1,
                        top: 0,
                        bottom: 0,
                        width: 2,
                        backgroundColor: "#f4f6fa",
                      }}
                    />
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
              Clean, Pitch and Timing call real server-side DSP and return a new local WAV. Regenerate, Harmony and Extend stay disabled until their dedicated AI jobs exist.
            </Text>
            <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 6 }}>
              <Button
                label={editingAction === "clean" ? "Cleaning…" : "Clean"}
                disabled={editingAction !== null || selectedClip.kind !== "audio"}
                onPress={() => void runClipAIAction("clean")}
              />
              <Button
                label={editingAction === "fix-pitch" ? "Fixing Pitch…" : "Fix Pitch"}
                disabled={editingAction !== null || selectedClip.kind !== "audio" || selectedClip.sourceArtifact !== "vocal"}
                onPress={() => void runClipAIAction("fix-pitch")}
              />
              <Button
                label={editingAction === "fix-timing" ? "Fixing Timing…" : "Fix Timing"}
                disabled={editingAction !== null || selectedClip.kind !== "audio" || selectedClip.sourceArtifact !== "vocal"}
                onPress={() => void runClipAIAction("fix-timing")}
              />
              <Button label="Regenerate" disabled onPress={() => undefined} />
              <Button label="Harmony" disabled onPress={() => undefined} />
              <Button label="Extend" disabled onPress={() => undefined} />
            </View>
          </View>
        </View>
      ) : (
        <Text selectable style={{ color: "#667183", fontSize: 11, lineHeight: 17 }}>
          Tap a clip to select it. Drag the clip body to move it, or drag the left/right handles to trim.
        </Text>
      )}

      {selectedClip?.kind === "midi" && selectedClip.sourcePart ? (
        <PianoRoll
          jobId={jobId}
          clip={selectedClip}
          bpm={config.bpm ?? 120}
          onSaved={(uri, part) => {
            onMusicPartsChange({ ...musicParts, [part]: uri });
            updateClip(selectedClip.id, { sourceUri: uri });
            onMessage(`✅ ${part} MIDI edits saved locally.`);
          }}
        />
      ) : null}

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#8f98aa", fontSize: 10, fontWeight: "900" }}>TRACK CONTROLS</Text>
        {TRACKS.map((track) => {
          const currentSetting = trackSettings[track.id] ?? defaultSetting(track.id);
          const midiUri = musicParts?.[track.id as keyof MusicPartFiles];

          return (
            <View key={track.id} style={{ width: "100%", borderRadius: 12, backgroundColor: "#0e1117", borderWidth: 1, borderColor: currentSetting.solo ? "#626d81" : "#252b38", padding: 9, gap: 7 }}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
                <View style={{ flex: 1 }}>
                  <Text selectable style={{ color: "#dce2ea", fontSize: 11, fontWeight: "800" }}>{track.label}</Text>
                  <Text selectable style={{ color: "#5e6878", fontSize: 8 }}>
                    {track.kind === "midi" ? "MIDI lane" : "Audio lane"} · {Math.round(currentSetting.volume * 100)}%
                  </Text>
                </View>
                <Button label="M" active={currentSetting.muted} onPress={() => updateTrack(track.id, { muted: !currentSetting.muted })} />
                <Button label="S" active={currentSetting.solo} onPress={() => updateTrack(track.id, { solo: !currentSetting.solo })} />
                <Button
                  label="VOL −"
                  disabled={currentSetting.volume <= 0}
                  onPress={() => updateTrack(track.id, { volume: Number(Math.max(0, currentSetting.volume - 0.1).toFixed(2)) })}
                />
                <Button
                  label="VOL +"
                  disabled={currentSetting.volume >= 1}
                  onPress={() => updateTrack(track.id, { volume: Number(Math.min(1, currentSetting.volume + 0.1).toFixed(2)) })}
                />
                {midiUri && track.kind === "midi" ? (
                  <Button label="Share MIDI" onPress={() => void shareMidi(midiUri, track.label)} />
                ) : null}
              </View>
            </View>
          );
        })}
      </View>

      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
        <Button
          label={renderingTimeline ? "Rendering WAV…" : "Render Timeline → WAV"}
          disabled={renderingTimeline}
          onPress={() => void renderCurrentTimeline()}
          active={!renderingTimeline}
        />
        <Button
          label={buildingParts ? "Generating MIDI…" : musicParts?.arrangement ? "Regenerate MIDI Parts" : "Generate MIDI Parts"}
          disabled={buildingParts}
          onPress={() => void createParts()}
        />
      </View>
    </View>
  );
}
