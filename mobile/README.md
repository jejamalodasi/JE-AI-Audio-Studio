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
GET /api/jobs/{job_id}/download/bundle

The song creation call uses multipart form data and returns a job ID with HTTP 202.

## Roadmap

- richer section controls and per-section duration editing
- full generation/master settings
- stem/mix controls
- share/export actions
- durable cloud projects + authentication
- multi-device sync
- EAS Update delivery
