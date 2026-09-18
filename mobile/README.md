# JE AI Audio Studio — Android Client

This directory is the first Expo/React Native control surface for the separate JE AI Audio Studio backend.

## Current flow

Vocal/audio file → upload to API → async Full AI Song job → poll status → download final master or project ZIP → local WAV playback.

The mobile app deliberately does **not** bundle the Python/AI models. Heavy MusicGen and DSP processing stays on the configured GPU/server backend.

## Configuration

Copy `.env.example` to `.env` and set:

EXPO_PUBLIC_API_URL=https://YOUR-SERVER

The Expo client only receives the public server URL. Do not place API secrets in EXPO_PUBLIC_ variables.

## Development

From the repository root:

cd mobile
npm install
npm run typecheck
npx expo start

Start with Expo Go for the first UI and network test. Expo SDK 57 is the current stable baseline documented by Expo; the app uses Expo Router, DocumentPicker, FileSystem and expo-audio.

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
GET /api/jobs/{job_id}/download/bundle

The song creation call uses multipart form data and returns a job ID with HTTP 202.

## Next mobile milestones

- richer section controls and per-section duration editing
- generation progress percentage and stage timeline
- project history
- offline job-state persistence
- stem/mix controls
- EAS Update delivery
- authentication once the backend is ready for multi-user hosting
