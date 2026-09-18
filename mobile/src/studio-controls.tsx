import { Pressable, Text, TextInput, View } from "react-native";

import type { SongConfig } from "./types";

const SECTION_PRESETS = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Hook", "Outro"];
const KEYS = ["Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SCALES = ["Auto", "Major", "Minor", "Dorian", "Phrygian", "Mixolydian"];
const CLEANUP = [
  { label: "Off", value: "none" as const },
  { label: "Basic", value: "basic" as const },
  { label: "Advanced", value: "advanced" as const },
];
const CONTINUITY = [
  { label: "Vocal Anchor", value: "vocal-anchor" as const },
  { label: "Chain", value: "chain" as const },
];

function Chip({
  label,
  active,
  onPress,
}: {
  label: string;
  active?: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={{
        borderRadius: 999,
        paddingHorizontal: 11,
        paddingVertical: 9,
        backgroundColor: active ? "#f4f6fa" : "#202633",
        borderWidth: 1,
        borderColor: active ? "#f4f6fa" : "#2d3543",
      }}
    >
      <Text style={{ color: active ? "#0b0d12" : "#cbd3df", fontSize: 12, fontWeight: "800" }}>
        {label}
      </Text>
    </Pressable>
  );
}

function Field({
  label,
  value,
  onChangeText,
  keyboardType = "default",
}: {
  label: string;
  value: string;
  onChangeText: (value: string) => void;
  keyboardType?: "default" | "numeric" | "decimal-pad";
}) {
  return (
    <View style={{ flex: 1, gap: 6, minWidth: 110 }}>
      <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
        {label}
      </Text>
      <TextInput
        value={value}
        onChangeText={onChangeText}
        keyboardType={keyboardType}
        placeholderTextColor="#626c7d"
        style={{
          minHeight: 44,
          borderRadius: 12,
          backgroundColor: "#0e1117",
          color: "#f5f7fb",
          paddingHorizontal: 12,
          borderWidth: 1,
          borderColor: "#252c39",
          fontWeight: "700",
        }}
      />
    </View>
  );
}

function MiniButton({
  label,
  onPress,
  disabled,
  danger,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={{
        minWidth: 38,
        minHeight: 34,
        borderRadius: 10,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#202633",
        borderWidth: 1,
        borderColor: danger ? "#5d3239" : "#303747",
        opacity: disabled ? 0.35 : 1,
      }}
    >
      <Text selectable style={{ color: danger ? "#ff9b9b" : "#d7deea", fontSize: 12, fontWeight: "900" }}>
        {label}
      </Text>
    </Pressable>
  );
}

export function StudioControls({
  config,
  setConfig,
}: {
  config: SongConfig;
  setConfig: React.Dispatch<React.SetStateAction<SongConfig>>;
}) {
  const totalSeconds = config.sections.reduce((sum, section) => sum + Math.max(0, section.duration_seconds), 0);
  const available = SECTION_PRESETS.filter(
    (name) => !config.sections.some((section) => section.name === name),
  );

  function updateSection(index: number, duration_seconds: number) {
    setConfig((current) => ({
      ...current,
      sections: current.sections.map((section, i) =>
        i === index ? { ...section, duration_seconds: Math.max(1, Math.min(30, duration_seconds)) } : section,
      ),
    }));
  }

  function moveSection(index: number, direction: -1 | 1) {
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= config.sections.length) return;
    setConfig((current) => {
      const sections = [...current.sections];
      const [moved] = sections.splice(index, 1);
      sections.splice(nextIndex, 0, moved);
      return { ...current, sections };
    });
  }

  function removeSection(index: number) {
    if (config.sections.length <= 1) return;
    setConfig((current) => ({
      ...current,
      sections: current.sections.filter((_, i) => i !== index),
    }));
  }

  function addSection(name: string) {
    setConfig((current) => ({
      ...current,
      sections: [...current.sections, { name, duration_seconds: name === "Chorus" ? 10 : 6 }],
    }));
  }

  return (
    <View style={{ gap: 14 }}>
      <View style={{ gap: 6 }}>
        <Text selectable style={{ color: "#f5f7fb", fontSize: 17, fontWeight: "800" }}>
          02 · Song Structure
        </Text>
        <Text selectable style={{ color: "#7f8899", fontSize: 12, lineHeight: 18 }}>
          Build the arrangement section-by-section. Total target: up to 90 seconds.
        </Text>
      </View>

      <View style={{ gap: 9 }}>
        {config.sections.map((section, index) => (
          <View
            key={`${section.name}-${index}`}
            style={{
              borderRadius: 14,
              backgroundColor: "#0e1117",
              borderWidth: 1,
              borderColor: "#262e3a",
              padding: 12,
              gap: 9,
            }}
          >
            <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <View
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: 9,
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "#202633",
                }}
              >
                <Text selectable style={{ color: "#bfc7d4", fontSize: 11, fontWeight: "900" }}>
                  {index + 1}
                </Text>
              </View>
              <View style={{ flex: 1, gap: 2 }}>
                <Text selectable style={{ color: "#edf1f7", fontWeight: "800" }}>
                  {section.name}
                </Text>
                <Text selectable style={{ color: "#687284", fontSize: 11 }}>
                  {Math.round(section.duration_seconds)} sec
                </Text>
              </View>
              <MiniButton label="↑" onPress={() => moveSection(index, -1)} disabled={index === 0} />
              <MiniButton
                label="↓"
                onPress={() => moveSection(index, 1)}
                disabled={index === config.sections.length - 1}
              />
              <MiniButton label="×" onPress={() => removeSection(index)} disabled={config.sections.length <= 1} danger />
            </View>

            <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
              <MiniButton label="−" onPress={() => updateSection(index, section.duration_seconds - 1)} />
              <View style={{ flex: 1, alignItems: "center" }}>
                <Text selectable style={{ color: "#f5f7fb", fontSize: 14, fontWeight: "900", fontVariant: ["tabular-nums"] }}>
                  {Math.round(section.duration_seconds)}s
                </Text>
              </View>
              <MiniButton
                label="+"
                onPress={() => updateSection(index, section.duration_seconds + 1)}
                disabled={totalSeconds >= 90}
              />
            </View>
          </View>
        ))}
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          ADD SECTION
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
          {available.map((name) => (
            <Chip key={name} label={name} onPress={() => addSection(name)} />
          ))}
          {available.length === 0 ? (
            <Text selectable style={{ color: "#687284", fontSize: 12 }}>
              All available section presets are already used.
            </Text>
          ) : null}
        </View>
      </View>

      <View
        style={{
          flexDirection: "row",
          justifyContent: "space-between",
          alignItems: "center",
          borderTopWidth: 1,
          borderTopColor: "#252c39",
          paddingTop: 11,
        }}
      >
        <Text selectable style={{ color: "#9da6b5", fontSize: 12 }}>
          Arrangement length
        </Text>
        <Text selectable style={{ color: totalSeconds > 90 ? "#ff9b9b" : "#f5f7fb", fontSize: 12, fontWeight: "900" }}>
          {Math.round(totalSeconds)} / 90 sec
        </Text>
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          VOCAL CLEANUP
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
          {CLEANUP.map((item) => (
            <Chip
              key={item.value}
              label={item.label}
              active={config.cleanup_mode === item.value}
              onPress={() => setConfig((current) => ({ ...current, cleanup_mode: item.value }))}
            />
          ))}
        </View>
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          MUSICAL GUIDE
        </Text>
        <View style={{ flexDirection: "row", gap: 8 }}>
          <Field
            label="BPM"
            value={config.bpm == null ? "" : String(config.bpm)}
            onChangeText={(value) =>
              setConfig((current) => ({
                ...current,
                bpm: value.trim() === "" ? null : Number(value),
              }))
            }
            keyboardType="numeric"
          />
          <Field
            label="Crossfade (sec)"
            value={String(config.crossfade_seconds)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, crossfade_seconds: Math.max(0, Math.min(2, next)) }));
            }}
            keyboardType="decimal-pad"
          />
        </View>

        <Text selectable style={{ color: "#687284", fontSize: 11 }}>
          Key
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 7 }}>
          {KEYS.map((value) => (
            <Chip
              key={value}
              label={value}
              active={(config.key ?? "Auto") === value}
              onPress={() => setConfig((current) => ({ ...current, key: value === "Auto" ? null : value }))}
            />
          ))}
        </View>

        <Text selectable style={{ color: "#687284", fontSize: 11 }}>
          Scale
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 7 }}>
          {SCALES.map((value) => (
            <Chip
              key={value}
              label={value}
              active={(config.scale ?? "Auto") === value}
              onPress={() => setConfig((current) => ({ ...current, scale: value === "Auto" ? null : value }))}
            />
          ))}
        </View>

        <Text selectable style={{ color: "#687284", fontSize: 11 }}>
          Continuity
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 7 }}>
          {CONTINUITY.map((item) => (
            <Chip
              key={item.value}
              label={item.label}
              active={config.continuity === item.value}
              onPress={() => setConfig((current) => ({ ...current, continuity: item.value }))}
            />
          ))}
        </View>
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          GENERATION
        </Text>
        <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
          <Field
            label="Guidance"
            value={String(config.guidance_scale)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, guidance_scale: Math.max(0, Math.min(10, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Temperature"
            value={String(config.temperature)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, temperature: Math.max(0.1, Math.min(2, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Top K"
            value={String(config.top_k)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, top_k: Math.max(0, Math.min(1000, Math.round(next))) }));
            }}
            keyboardType="numeric"
          />
          <Field
            label="Seed"
            value={String(config.seed)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, seed: Math.round(next) }));
            }}
            keyboardType="numeric"
          />
        </View>
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          MIX + MASTER
        </Text>
        <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
          <Field
            label="Vocal dB"
            value={String(config.vocal_gain_db)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, vocal_gain_db: Math.max(-12, Math.min(6, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Music dB"
            value={String(config.music_gain_db)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, music_gain_db: Math.max(-12, Math.min(6, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Compression"
            value={String(config.compression_ratio)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, compression_ratio: Math.max(1, Math.min(8, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Saturation"
            value={String(config.saturation)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, saturation: Math.max(0, Math.min(1, next)) }));
            }}
            keyboardType="decimal-pad"
          />
          <Field
            label="Target Peak"
            value={String(config.target_peak)}
            onChangeText={(value) => {
              const next = Number(value);
              if (Number.isFinite(next)) setConfig((current) => ({ ...current, target_peak: Math.max(0.5, Math.min(1, next)) }));
            }}
            keyboardType="decimal-pad"
          />
        </View>
      </View>

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          ENGINE
        </Text>
        <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8 }}>
          {(["auto", "cuda", "cpu"] as const).map((value) => (
            <Chip
              key={value}
              label={value.toUpperCase()}
              active={config.device === value}
              onPress={() => setConfig((current) => ({ ...current, device: value }))}
            />
          ))}
        </View>
      </View>
    </View>
  );
}
