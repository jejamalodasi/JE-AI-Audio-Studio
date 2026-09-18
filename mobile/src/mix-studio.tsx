import { useAudioPlayer, useAudioPlayerStatus } from "expo-audio";
import * as Sharing from "expo-sharing";
import { Pressable, Text, View } from "react-native";

import { remixSongJob, downloadArtifact } from "./api";

type Artifact = {
  label: string;
  uri?: string;
  kind: "final" | "vocal" | "backing" | "remix";
};

function formatTime(value: number) {
  if (!Number.isFinite(value) || value < 0) return "0:00";
  const total = Math.floor(value);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function SmallButton({
  label,
  onPress,
  disabled,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={{
        minHeight: 36,
        minWidth: 52,
        borderRadius: 10,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#202633",
        borderWidth: 1,
        borderColor: "#303747",
        opacity: disabled ? 0.4 : 1,
      }}
    >
      <Text selectable style={{ color: "#dce2ec", fontSize: 11, fontWeight: "800" }}>
        {label}
      </Text>
    </Pressable>
  );
}

function TrackStrip({
  artifact,
  volume,
  muted,
  active,
  onPlay,
  onVolume,
  onMute,
}: {
  artifact: Artifact;
  volume: number;
  muted: boolean;
  active: boolean;
  onPlay: () => void;
  onVolume: (value: number) => void;
  onMute: () => void;
}) {
  const player = useAudioPlayer(artifact.uri || null, { updateInterval: 500 });
  const status = useAudioPlayerStatus(player);

  const setPlayerVolume = (value: number) => {
    player.volume = value;
  };

  const playing = status.playing && active;

  return (
    <View
      style={{
        borderRadius: 14,
        padding: 12,
        gap: 10,
        backgroundColor: "#0e1117",
        borderWidth: 1,
        borderColor: playing ? "#596274" : "#262e3a",
      }}
    >
      <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
        <View
          style={{
            width: 34,
            height: 34,
            borderRadius: 10,
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#202633",
          }}
        >
          <Text selectable style={{ color: "#dce2ec", fontSize: 11, fontWeight: "900" }}>
            {artifact.label.slice(0, 2).toUpperCase()}
          </Text>
        </View>

        <View style={{ flex: 1, gap: 3 }}>
          <Text selectable style={{ color: "#edf1f7", fontWeight: "800" }}>
            {artifact.label}
          </Text>
          <Text selectable style={{ color: "#687284", fontSize: 11 }}>
            {artifact.uri ? `${formatTime(status.currentTime)} / ${formatTime(status.duration)}` : "Save this stem to enable playback"}
          </Text>
        </View>

        <SmallButton
          label={playing ? "Pause" : "Play"}
          disabled={!artifact.uri}
          onPress={() => {
            if (!artifact.uri) return;
            onPlay();
            if (playing) player.pause();
            else player.play();
          }}
        />
      </View>

      {artifact.uri ? (
        <View style={{ gap: 8 }}>
          <View
            style={{
              height: 5,
              borderRadius: 5,
              overflow: "hidden",
              backgroundColor: "#252b38",
            }}
          >
            <View
              style={{
                width: `${status.duration > 0 ? Math.min(100, (status.currentTime / status.duration) * 100) : 0}%`,
                height: "100%",
                backgroundColor: "#dce2ec",
              }}
            />
          </View>

          <View style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <Text selectable style={{ color: "#7f8899", width: 58, fontSize: 11 }}>
              Vol {Math.round(volume * 100)}%
            </Text>
            <SmallButton
              label="−"
              onPress={() => {
                const next = Math.max(0, Number((volume - 0.1).toFixed(2)));
                onVolume(next);
                setPlayerVolume(muted ? 0 : next);
              }}
            />
            <View
              style={{
                height: 6,
                flex: 1,
                borderRadius: 6,
                backgroundColor: "#252b38",
                overflow: "hidden",
              }}
            >
              <View
                style={{
                  width: `${Math.round(volume * 100)}%`,
                  height: "100%",
                  backgroundColor: "#bfc7d4",
                }}
              />
            </View>
            <SmallButton
              label="+"
              onPress={() => {
                const next = Math.min(1, Number((volume + 0.1).toFixed(2)));
                onVolume(next);
                setPlayerVolume(muted ? 0 : next);
              }}
            />
            <SmallButton
              label={muted ? "Unmute" : "Mute"}
              onPress={() => {
                onMute();
                setPlayerVolume(muted ? volume : 0);
              }}
            />
          </View>
        </View>
      ) : null}
    </View>
  );
}

export function MixStudio({
  jobId,
  artifacts,
  vocalGain,
  backingGain,
  targetPeak,
  compressionRatio,
  saturation,
  onRemixSaved,
  onMessage,
}: {
  jobId: string;
  artifacts: {
    final?: string;
    vocal?: string;
    backing?: string;
    remix?: string;
  };
  vocalGain: number;
  backingGain: number;
  targetPeak: number;
  compressionRatio: number;
  saturation: number;
  onRemixSaved: (uri: string) => void;
  onMessage: (message: string) => void;
}) {
  const [volumes, setVolumes] = require("react").useState({
    vocal: 0.9,
    backing: 0.75,
    final: 1,
    remix: 1,
  });
  const [muted, setMuted] = require("react").useState({
    vocal: false,
    backing: false,
    final: false,
    remix: false,
  });
  const [active, setActive] = require("react").useState<string | null>(null);
  const [mixing, setMixing] = require("react").useState(false);

  const [currentVocalGain, setCurrentVocalGain] = require("react").useState(vocalGain);
  const [currentBackingGain, setCurrentBackingGain] = require("react").useState(backingGain);
  const [currentTargetPeak, setCurrentTargetPeak] = require("react").useState(targetPeak);
  const [currentCompression, setCurrentCompression] = require("react").useState(compressionRatio);
  const [currentSaturation, setCurrentSaturation] = require("react").useState(saturation);

  async function applyMix() {
    setMixing(true);
    onMessage("Creating a new remix from the saved vocal + backing stems…");
    try {
      await remixSongJob(jobId, {
        vocal_gain_db: currentVocalGain,
        backing_gain_db: currentBackingGain,
        target_peak: currentTargetPeak,
        compression_ratio: currentCompression,
        saturation: currentSaturation,
      });

      const uri = await downloadArtifact(jobId, "remix", "wav");
      onRemixSaved(uri);
      onMessage("✅ New remix saved locally.");
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Could not create remix.");
    } finally {
      setMixing(false);
    }
  }

  async function shareFile(uri: string, label: string) {
    try {
      const available = await Sharing.isAvailableAsync();
      if (!available) {
        onMessage("Sharing is not available on this device.");
        return;
      }
      await Sharing.shareAsync(uri, {
        dialogTitle: `Share ${label}`,
        mimeType: "audio/wav",
      });
    } catch (error) {
      onMessage(error instanceof Error ? error.message : "Could not share the file.");
    }
  }

  function numericStep(
    current: number,
    setter: (value: number) => void,
    step: number,
    min: number,
    max: number,
  ) {
    setter(Math.max(min, Math.min(max, Number((current + step).toFixed(2)))));
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
          06 · Stem / Mix Studio
        </Text>
        <Text selectable style={{ color: "#7f8899", fontSize: 12, lineHeight: 18 }}>
          Preview the generated tracks separately, adjust the remix bus and render a new WAV without regenerating the AI arrangement.
        </Text>
      </View>

      <TrackStrip
        artifact={{ label: "Final Master", uri: artifacts.final, kind: "final" }}
        volume={volumes.final}
        muted={muted.final}
        active={active === "final"}
        onPlay={() => setActive("final")}
        onVolume={(value) => setVolumes((current: typeof volumes) => ({ ...current, final: value }))}
        onMute={() => setMuted((current: typeof muted) => ({ ...current, final: !current.final }))}
      />
      <TrackStrip
        artifact={{ label: "Cleaned Vocal", uri: artifacts.vocal, kind: "vocal" }}
        volume={volumes.vocal}
        muted={muted.vocal}
        active={active === "vocal"}
        onPlay={() => setActive("vocal")}
        onVolume={(value) => setVolumes((current: typeof volumes) => ({ ...current, vocal: value }))}
        onMute={() => setMuted((current: typeof muted) => ({ ...current, vocal: !current.vocal }))}
      />
      <TrackStrip
        artifact={{ label: "AI Backing", uri: artifacts.backing, kind: "backing" }}
        volume={volumes.backing}
        muted={muted.backing}
        active={active === "backing"}
        onPlay={() => setActive("backing")}
        onVolume={(value) => setVolumes((current: typeof volumes) => ({ ...current, backing: value }))}
        onMute={() => setMuted((current: typeof muted) => ({ ...current, backing: !current.backing }))}
      />
      {artifacts.remix ? (
        <TrackStrip
          artifact={{ label: "Latest Remix", uri: artifacts.remix, kind: "remix" }}
          volume={volumes.remix}
          muted={muted.remix}
          active={active === "remix"}
          onPlay={() => setActive("remix")}
          onVolume={(value) => setVolumes((current: typeof volumes) => ({ ...current, remix: value }))}
          onMute={() => setMuted((current: typeof muted) => ({ ...current, remix: !current.remix }))}
        />
      ) : null}

      <View style={{ gap: 8 }}>
        <Text selectable style={{ color: "#7f8899", fontSize: 11, fontWeight: "700" }}>
          REMIX BUS
        </Text>
        <Text selectable style={{ color: "#687284", fontSize: 11 }}>
          Vocal / backing gain and master-bus settings affect the new rendered remix. Playback volume above is monitor-only.
        </Text>

        <View style={{ gap: 8 }}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
            <Text selectable style={{ color: "#cbd3df", fontSize: 12 }}>Vocal gain</Text>
            <Text selectable style={{ color: "#f5f7fb", fontWeight: "900" }}>{currentVocalGain.toFixed(1)} dB</Text>
          </View>
          <View style={{ flexDirection: "row", gap: 8 }}>
            <SmallButton label="−" onPress={() => numericStep(currentVocalGain, setCurrentVocalGain, -1, -12, 6)} />
            <SmallButton label="0" onPress={() => setCurrentVocalGain(0)} />
            <SmallButton label="+" onPress={() => numericStep(currentVocalGain, setCurrentVocalGain, 1, -12, 6)} />
          </View>
        </View>

        <View style={{ gap: 8 }}>
          <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
            <Text selectable style={{ color: "#cbd3df", fontSize: 12 }}>Backing gain</Text>
            <Text selectable style={{ color: "#f5f7fb", fontWeight: "900" }}>{currentBackingGain.toFixed(1)} dB</Text>
          </View>
          <View style={{ flexDirection: "row", gap: 8 }}>
            <SmallButton label="−" onPress={() => numericStep(currentBackingGain, setCurrentBackingGain, -1, -12, 6)} />
            <SmallButton label="0" onPress={() => setCurrentBackingGain(0)} />
            <SmallButton label="+" onPress={() => numericStep(currentBackingGain, setCurrentBackingGain, 1, -12, 6)} />
          </View>
        </View>

        <View style={{ flexDirection: "row", gap: 8 }}>
          <SmallButton label="Peak −" onPress={() => numericStep(currentTargetPeak, setCurrentTargetPeak, -0.05, 0.5, 1)} />
          <SmallButton label="Peak +" onPress={() => numericStep(currentTargetPeak, setCurrentTargetPeak, 0.05, 0.5, 1)} />
          <SmallButton label="Comp −" onPress={() => numericStep(currentCompression, setCurrentCompression, -0.5, 1, 8)} />
          <SmallButton label="Comp +" onPress={() => numericStep(currentCompression, setCurrentCompression, 0.5, 1, 8)} />
        </View>

        <View style={{ flexDirection: "row", gap: 8 }}>
          <SmallButton label="Sat −" onPress={() => numericStep(currentSaturation, setCurrentSaturation, -0.05, 0, 1)} />
          <SmallButton label="Sat +" onPress={() => numericStep(currentSaturation, setCurrentSaturation, 0.05, 0, 1)} />
          <Text selectable style={{ color: "#687284", fontSize: 11, alignSelf: "center" }}>
            Peak {currentTargetPeak.toFixed(2)} · Comp {currentCompression.toFixed(1)} · Sat {currentSaturation.toFixed(2)}
          </Text>
        </View>
      </View>

      <SmallButton
        label={mixing ? "Rendering Remix…" : "Render New Remix"}
        disabled={mixing}
        onPress={() => void applyMix()}
      />

      {artifacts.remix ? (
        <SmallButton label="Share Latest Remix" onPress={() => void shareFile(artifacts.remix!, "latest remix")} />
      ) : null}

      <View style={{ flexDirection: "row", gap: 8, flexWrap: "wrap" }}>
        {(["final", "vocal", "backing"] as const).map((kind) => {
          const uri = artifacts[kind];
          if (!uri) return null;
          const label = kind === "final" ? "Final" : kind === "vocal" ? "Vocal" : "Backing";
          return (
            <SmallButton
              key={kind}
              label={`Share ${label}`}
              onPress={() => void shareFile(uri, label)}
            />
          );
        })}
      </View>
    </View>
  );
}
