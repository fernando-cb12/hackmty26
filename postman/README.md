# Postman contract tests

The collection tests the current `scripts/example_server.py` implementation and
the same protocol used by `scripts/check_endpoint.py`.

## Protocol under test

```http
POST /detect
Content-Type: application/json
```

```json
{
  "call_id": "call_76856257e3ef",
  "audio_base64": "<base64 of the complete WAV file>",
  "sample_rate": 8000,
  "channels": 2
}
```

The WAV must be stereo, 8 kHz, 16-bit PCM. A successful response is HTTP 200:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
```

`is_synthetic` must be a boolean. `confidence` must be a number from 0 to 1.
Every request must finish within 30 seconds.

## Prepare a real call without committing audio

Postman's sandbox cannot read an arbitrary local WAV into a raw JSON request.
Generate a local environment that contains one WAV encoded as base64:

```bash
python3 postman/build_environment.py audio/call_76856257e3ef.wav
```

To test the stable Cloudflare hostname instead:

```bash
python3 postman/build_environment.py \
  audio/call_76856257e3ef.wav \
  --base-url https://demo.example.com
```

The generated environment is ignored by Git because it contains challenge
audio. Do not commit or redistribute it.

## Run the collection

1. Start the API from the repository root:

   ```bash
   .venv/bin/python scripts/example_server.py --port 8000
   ```

2. Import both files into Postman:

   - `postman/HackMTY26.postman_collection.json`
   - `postman/HackMTY26.generated.postman_environment.json`

3. Select **HackMTY 2026 - Generated Local Test** as the active environment.
4. Run the complete collection.

The first request sends a real call and validates the successful judge response.
The remaining requests verify that the server rejects an invalid sample rate,
channel count, malformed base64, missing audio, and a wrong route.
