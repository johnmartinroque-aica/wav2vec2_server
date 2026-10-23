# wav2vec2 Pronunciation Server

A local pronunciation checker for English learners. You send a recording and the sentence the learner was asked to say, and the server returns:

- an overall score and a letter grade
- a score for each word
- the sounds that were missed, swapped, or added, with tips on how to make each sound
- optional friendly coaching text for young learners, written by a local [Ollama](https://ollama.com) model

Everything runs on your own computer. After the model downloads once, no audio leaves the machine.

## How it works

1. **Recognition:** [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) turns the audio into a sequence of IPA phonemes.
2. **Reference:** [eSpeak NG](https://github.com/espeak-ng/espeak-ng) (through `phonemizer`) converts the target sentence into the phonemes it should contain (en-US).
3. **Alignment and scoring:** the two sequences are lined up. The overall score is how closely they match (0–100). Each word's score is the percentage of its sounds that were said correctly.
4. **Explanations:** `phonemes.py` turns each mismatch into plain English, for example *you said "d" instead of "th"*, and adds a tip on mouth position.
5. **Coaching (optional):** `feedback.py` sends the top 2 issues to a local Ollama model. The model rewrites them as short, encouraging advice for a child. It only rewords the issues it is given and never judges the audio itself.

| File             | Purpose                                                        |
| ---------------- | -------------------------------------------------------------- |
| `server.py`      | FastAPI HTTP server (`/analyze`, `/health`)                    |
| `pronounce.py`   | Core analysis, plus a command-line microphone practice tool    |
| `phonemes.py`    | Plain-English sound names, tips, and common mix-up advice      |
| `scoring.py`     | Grade bands (A–D)                                              |
| `feedback.py`    | Child-friendly coaching through Ollama                         |

## Requirements

- Python 3.10+
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases). On Windows, the default install path `C:\Program Files\eSpeak NG` is detected automatically. Elsewhere, set `PHONEMIZER_ESPEAK_LIBRARY`.
- [ffmpeg](https://ffmpeg.org/) on `PATH`, needed only for non-WAV uploads (webm, ogg, mp4, mp3)
- [Ollama](https://ollama.com), needed only for the `feedback` text
- About 1.2 GB of disk space for the model (downloaded on first run into `~/.cache/huggingface`)

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install librosa   # only needed to resample WAV files that aren't 16 kHz in the CLI
```

Optional, for the coaching feedback:

```bash
ollama pull llama3.2:1b
ollama serve
```

## Run the server

```bash
python server.py
```

- API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Health check: `GET /health` returns `{"ok": true}`

### Configuration

| Variable          | Meaning                                                          | Default                  |
| ----------------- | ---------------------------------------------------------------- | ------------------------ |
| `HOST`, `PORT`    | Where to listen                                                  | `127.0.0.1`, `8000`      |
| `ALLOWED_ORIGINS` | Comma-separated sites allowed to call the API (CORS)             | `*`                      |
| `API_KEY`         | If set, requests must send the header `X-API-Key: <value>`       | not set                  |
| `OLLAMA_URL`      | Ollama server address                                            | `http://127.0.0.1:11434` |
| `OLLAMA_MODEL`    | Model used for feedback                                          | `llama3.2:1b`            |
| `OLLAMA_TIMEOUT`  | Seconds to wait for Ollama                                       | `30`                     |

## API

### `POST /analyze`

`multipart/form-data` fields:

- `audio`: the recording (WAV preferred, max 10 MB)
- `text`: the word or sentence the learner was asked to say

```bash
curl -F "audio=@recording.wav" -F "text=Rock the lock, then lock the rock." \
     http://127.0.0.1:8000/analyze
```

Response (shortened):

```json
{
  "ok": true,
  "score": 87,
  "grade": { "letter": "B", "verdict": "Good", "color": "#65a30d" },
  "allWordsCorrect": false,
  "expected": { "phonemes": ["ɹ", "ɑ", "k", "..."], "soundsLike": "r-ah-k-..." },
  "heard":    { "phonemes": ["r", "ɔ", "k", "..."], "soundsLike": "r-aw-k-..." },
  "words": [
    { "word": "Rock", "score": 67, "phonemes": ["ɹ", "ɑ", "k"], "soundsLike": "r-ah-k" }
  ],
  "issues": [
    { "word": "Rock", "message": "you said \"aw\" ... instead of \"ah\" ...", "tip": null }
  ],
  "feedback": "You're doing great! 😊 In \"Rock\", ..."
}
```

- `feedback` is `null` if Ollama isn't running. Everything else still works.
- If the recording is too short or too quiet:
  `{"ok": false, "error": "too_short_or_quiet", "message": "..."}`

### Grades

| Grade | Score | Meaning                                                |
| ----- | ----- | ------------------------------------------------------ |
| A     | ≥ 90  | Clear, native-like; almost every sound recognised      |
| B     | ≥ 75  | Easy to understand; a few sounds slightly off          |
| C     | ≥ 55  | Understandable, but several sounds wrong or missing    |
| D     | < 55  | Hard to understand; many sounds wrong or missing       |

## Command-line practice tool

```bash
python pronounce.py                                  # practise the built-in tongue twisters
python pronounce.py "The weather is nice today"      # practise your own sentence
python pronounce.py --file take.wav "Rock the lock"  # score a WAV file
python pronounce.py --list-devices                   # list microphones
python pronounce.py --device 2 --save                # choose a mic and save each take
```


## Hosting note

A website can reach `127.0.0.1` only on the computer running the server. If the site is served over HTTPS, host this server over HTTPS too, and set `ALLOWED_ORIGINS` to the site's address.
