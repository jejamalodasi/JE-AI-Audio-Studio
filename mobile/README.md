# JE AI Audio Studio — Android Client

This directory is the first Expo/React Native control surface for the separate JE AI Audio Studio backend.

## Current flow

Vocal/audio file → persistent local source copy → upload to API → async Full AI Song job → poll status → save final/stems/project bundle → local project history.

The mobile app deliberately does **not** bundle the Python/AI models. Heavy MusicGen and DSP processing stays on the configured GPU/server backend.

## Saving model

The mobile client is **local-first** right now. A login is **not required**.

- Project settings, source metadata, job state and local artifact paths are stored with Expo SQLite's key-value storage.
- Selected source audio is copied into the app's persistent Documents storage instead of relying only on temporary cache.
- Downloaded final WAV, cleaned vocal, backing and project ZIP are stored in a persistent Documents/exports area.
- The app restores the latest local draft on startup and can resume polling an active queued/running job.
- Up to 25 recent projects are kept in local history.

This survives normal app restarts. It does not provide automatic cross-device sync, and uninstalling the app can remove its local database/files. Cloud projects and multi-device sync will require authentication plus durable server storage.

Expo documents expo-sqlite as persistent across app restarts, and its kv-store API is a SQLite-backed key-value store suitable for this local state.

## Configuration

Copy .env.example to .env and set:

EXPO_PUBLIC_API_URL=https://YOUR-SERVER

The Expo client only receives the public server URL. Do not place API secrets in EXPO_PUBLIC_ variables.

## Development

From the repository root:

cd mobile
npm install
npm run typecheck
npx expo start

Expo SDK 57 is the current stable baseline documented by Expo, and the app uses Expo Router, DocumentPicker, FileSystem, SQLite and expo-audio.

## Android build

The included eas.json has a preview profile that builds an internal-distribution APK:

cd mobile
eas build --platform android --profile preview

The backend must be reachable from the device running the app. For a public deployment, point EXPO_PUBLIC_API_URL at the deployed HTTPS API.

## API contract

The client currently uses:

GET /health
POST /api/jobs/song
GET /api/jobs/{job_id}
GET /api/jobs/{job_id}/download/final
GET /api/jobs/{job_id}/download/vocal
GET /api/jobs/{job_id}/download/backing
POST /api/jobs/{job_id}/remix
GET /api/jobs/{job_id}/download/remix
GET /api/jobs/{job_id}/download/bundle

The song creation call uses multipart form data and returns a job ID with HTTP 202.

## Current Studio controls

The Android client now exposes the main Full AI Song configuration directly in the UI:

- AI style prompt
- section add/remove/reorder
- per-section duration control with a 90-second guard
- vocal cleanup: Off / Basic / Advanced
- BPM, key, scale and continuity
- crossfade
- generation guidance, temperature, Top-K, Top-P and seed
- vocal/music gain
- compression, saturation and target peak
- engine selection: Auto / CUDA / CPU

The selected configuration is part of the locally saved project, so opening a saved project restores its studio settings.

## Stem / Mix Studio

Completed jobs can be opened in the mobile Mix Studio to preview the final master, cleaned vocal, AI backing and latest remix independently. Each local player has monitor volume and mute controls.

The Remix Bus can re-render the completed vocal + backing stems with new vocal gain, backing gain, target peak, compression ratio and saturation settings. This does not regenerate the AI backing, so remix iterations are much lighter than a full song generation.

Native file sharing is provided through expo-sharing on Android/iOS.

## Multi-Track Timeline

Completed jobs can now open a timeline with:

- Vocal and AI Backing audio lanes
- Melody, Chords, Bass, Drums and Rhythm MIDI lanes
- section-based visual clip blocks using the same arrangement structure
- per-track monitor volume, mute and solo state saved with the local project
- one-tap generation of independent MIDI part files
- MIDI sharing from the device

The current musical-part engine exports **real MIDI files**, not synthesized audio stems. This is intentional: it keeps the current implementation honest and lightweight. Audio synthesis for those separate parts needs a dedicated instrument/sampler or a separately licensed generative backend.

The API adds:

POST /api/jobs/{job_id}/parts
GET  /api/jobs/{job_id}/download/melody
GET  /api/jobs/{job_id}/download/chords
GET  /api/jobs/{job_id}/download/bass
GET  /api/jobs/{job_id}/download/drums
GET  /api/jobs/{job_id}/download/rhythm
GET  /api/jobs/{job_id}/download/arrangement

## Roadmap

- true multi-stem audio synthesis/separation
- richer mix automation
- durable cloud projects + authentication
- multi-device sync
- EAS Update delivery
