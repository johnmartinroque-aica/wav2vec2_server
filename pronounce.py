"""Live pronunciation checker: speak into the microphone, get a phoneme-level score.

Usage:
    python pronounce.py                       # practice the built-in sentences
    python pronounce.py "The weather is nice today"
    python pronounce.py --file recording.wav "The weather is nice today"
    python pronounce.py --list-devices
"""
import argparse
import os
import queue
import sys
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"

PRACTICE_SENTENCES = [
    "Rock the lock, then lock the rock.",
    "The clown wore a crown in the crowd.",
    "I bet the vet will be back by seven.",
    "Curb the curve before the next curb.",
    "She will sip her soup on the ship.",
    "Sue lost one shoe, so Sue bought a new shoe.",
    "A bug bit the boy, the boy beat the bug.",
    "I think the ship will sink, I think.",
    "Though the dough won't thaw, to the oven it goes.",
    "Show the proof to prove that you have proof.",
    "Rory rented what Larry lent.",
    "Frame the flame before the flame flickers.",
    "Vote for the boat, then boat to the vote.",
    "The rebels revel by the river.",
    "See the water seep near the sleeping sheep.",
    "The shy girl let out a big sad sigh.",
    "The sheep will slip, then the sheep will sleep.",
    "The hen then ran out of the den.",
    "Watch the sun rise as you eat rice.",
    "The dog dashed down the dock.",
]

# On Windows, phonemizer needs to be told where the eSpeak NG library lives.
if sys.platform == "win32" and "PHONEMIZER_ESPEAK_LIBRARY" not in os.environ:
    _dll = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    if os.path.exists(_dll):
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _dll

import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from phonemizer import phonemize  # noqa: E402
from phonemizer.separator import Separator  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: E402

from phonemes import explain, respell  # noqa: E402
from scoring import grade  # noqa: E402

# Marks that differ between the model output and espeak but don't change
# whether a sound was pronounced correctly.
_STRIP = str.maketrans("", "", "ˈˌːˑ‿")

# Sounds that differ mainly in stress, which the model doesn't tell apart
# reliably. Each maps to the one sound the whole group counts as.
_SAME = {
    "ʌ": "ə", "ɐ": "ə",  # cup / sofa / about -> "uh"
    "ɚ": "ɜ", "ɝ": "ɜ",  # butter / bird -> "er"
    "ᵻ": "ɪ", "ɨ": "ɪ",  # roses (weak) -> "i"
    "ɾ": "t",            # flapped t in "water"
}


def normalize(phone):
    phone = phone.translate(_STRIP)
    return _SAME.get(phone, phone)


def load_model():
    print("Loading model (first run downloads ~1.2 GB)...")
    proc = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    return proc, model


def record_from_mic(device=None):
    """Record until the user presses Enter. Returns float32 mono audio at 16 kHz."""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    input("Press Enter and start speaking...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        device=device, callback=callback):
        input("  Recording... press Enter when done.")
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:, 0]


def trim_silence(audio, threshold=0.01, pad=0.2):
    """Cut leading/trailing silence so the model doesn't hallucinate on noise."""
    loud = np.where(np.abs(audio) > threshold)[0]
    if len(loud) == 0:
        return audio
    p = int(pad * SAMPLE_RATE)
    return audio[max(0, loud[0] - p): loud[-1] + p]


def load_wav(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa  # only needed for resampling files
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def emissions(proc, model, audio):
    """Per-frame log-probabilities over the model's vocabulary, shape (frames, vocab)."""
    inputs = proc(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        return model(**inputs).logits[0].log_softmax(-1)


def _skip_ids(tokenizer):
    """Vocab ids that aren't sounds: blank/special tokens and the word delimiter."""
    skip = set(tokenizer.all_special_ids)
    delim = tokenizer.get_vocab().get(tokenizer.word_delimiter_token or "")
    if delim is not None:
        skip.add(delim)
    return skip


def phone_classes(tokenizer):
    """Group vocab ids by normalized phone, so e.g. 'iː' and 'i' count as one sound."""
    skip = _skip_ids(tokenizer)
    classes = {}
    for tok, i in tokenizer.get_vocab().items():
        p = normalize(tok)
        if i not in skip and p:
            classes.setdefault(p, []).append(i)
    return classes


def recognize_phones(emission, tokenizer):
    """Greedy CTC decode. Returns [(phone, frame), ...] - what the model heard, and when."""
    skip = _skip_ids(tokenizer)
    out, prev = [], None
    for frame, i in enumerate(emission.argmax(-1).tolist()):
        if i != prev and i not in skip:
            p = normalize(tokenizer.convert_ids_to_tokens(i))
            if p:
                out.append((p, frame))
        prev = i
    return out


def align(emission, tokenizer, phones):
    """Force-align the expected phones to the audio with torchaudio's CTC aligner.

    Returns [(start_frame, end_frame, confidence), ...] - one per phone, confidence being
    the model's average probability (0-1) that the expected sound is there. Phones missing
    from the vocab get None. Returns None if the audio is too short for the sentence.
    """
    classes = phone_classes(tokenizer)
    known = [i for i, p in enumerate(phones) if p in classes]
    if not known:
        return None
    # Only blank and the expected sounds matter to the aligner, so build a small emission
    # with one column per sound (summing variants like 'i'/'iː') and blank at column 0.
    uniq = list(dict.fromkeys(phones[i] for i in known))
    cols = [emission[:, tokenizer.pad_token_id]]
    cols += [emission[:, classes[p]].logsumexp(-1) for p in uniq]
    targets = torch.tensor([[uniq.index(phones[i]) + 1 for i in known]], dtype=torch.int32)
    try:
        labels, scores = AF.forced_align(torch.stack(cols, -1).unsqueeze(0), targets, blank=0)
    except RuntimeError:  # more sounds than audio frames
        return None
    spans = AF.merge_tokens(labels[0], scores[0].exp())
    result = [None] * len(phones)
    for i, s in zip(known, spans):
        result[i] = (s.start, s.end, s.score)
    return result


def expected_phones_per_word(text):
    """Return [(word, [phones...]), ...] using espeak."""
    words = text.split()
    phonemized = phonemize(words, language="en-us", backend="espeak",
                           separator=Separator(phone=" ", word="", syllable=""),
                           strip=True, with_stress=False, njobs=1)
    result = []
    for word, ph in zip(words, phonemized):
        phones = [normalize(p) for p in ph.split() if normalize(p)]
        result.append((word.strip(".,!?;:\"'"), phones))
    return result


def analyze(proc, model, audio, text):
    """Score audio against text. Returns a dict, or None if the audio is unusable."""
    audio = trim_silence(audio)
    if len(audio) < SAMPLE_RATE * 0.3:
        return None

    emission = emissions(proc, model, audio)
    heard = recognize_phones(emission, proc.tokenizer)
    spoken = [p for p, _ in heard]
    words = expected_phones_per_word(text)

    expected, owner = [], []  # owner[i] = index of the word that phone i belongs to
    for wi, (_, phones) in enumerate(words):
        expected.extend(phones)
        owner.extend([wi] * len(phones))

    sm = SequenceMatcher(None, expected, spoken, autojunk=False)
    correct = [False] * len(expected)
    issues = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for i in range(i1, i2):
                correct[i] = True
        else:
            word = words[owner[min(i1, len(owner) - 1)]][0] if owner else "?"
            issues.append((op, word, expected[i1:i2], spoken[j1:j2]))

    word_scores = []
    for wi, (word, phones) in enumerate(words):
        idx = [i for i, o in enumerate(owner) if o == wi]
        ws = 100 * sum(correct[i] for i in idx) / len(idx) if idx else 100
        word_scores.append((word, phones, ws))

    spans = align(emission, proc.tokenizer, expected)
    seconds_per_frame = len(audio) / SAMPLE_RATE / emission.shape[0]
    timing = None
    if spans:
        timing = []
        for wi in range(len(words)):
            ws = [spans[i] for i, o in enumerate(owner) if o == wi and spans[i]]
            timing.append({
                "start": ws[0][0] * seconds_per_frame if ws else None,
                "end": ws[-1][1] * seconds_per_frame if ws else None,
                "confidence": 100 * sum(s[2] for s in ws) / len(ws) if ws else None,
            })
    confs = [s[2] for s in spans or [] if s]

    return {
        "score": sm.ratio() * 100,
        "expected": expected,
        "spoken": spoken,
        "words": word_scores,
        "issues": issues,
        # From forced alignment (None if it couldn't align):
        "confidence": 100 * sum(confs) / len(confs) if confs else None,
        "timing": timing,  # per word: start/end seconds, confidence 0-100
        "heard_words": heard_per_word(heard, spans, owner, len(words), sm),
    }


def heard_per_word(heard, spans, owner, n_words, sm):
    """Split what the model heard into the sentence's words.

    With alignment, each heard sound goes to the word that had started by then, so an
    extra vowel ("milk" -> "miruku") stays with the word it follows. Without it, fall
    back to the text match between expected and heard sounds.
    """
    groups = [[] for _ in range(n_words)]
    if not owner:
        return groups
    if spans:
        starts = [None] * n_words
        for i, s in enumerate(spans):
            if s and starts[owner[i]] is None:
                starts[owner[i]] = s[0]
        for phone, frame in heard:
            wi = max((w for w, st in enumerate(starts) if st is not None and st <= frame),
                     default=0)
            groups[wi].append(phone)
        return groups
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        i = i1 - 1 if op == "insert" and i1 > 0 else i1
        groups[owner[min(i, len(owner) - 1)]].extend(p for p, _ in heard[j1:j2])
    return groups


def assess(proc, model, audio, text):
    r = analyze(proc, model, audio, text)
    if r is None:
        print("  Didn't catch that - recording was too short or too quiet.\n")
        return None

    score = r["score"]
    g = grade(score)
    print(f"\n  Expected: {respell(r['expected'])}")
    print(f"  Heard:    {respell(r['spoken'])}")
    print(f"\n  Overall score: {score:.0f}/100  Grade {g.letter} - {g.verdict}\n")

    print("  Per word:")
    for word, phones, ws in r["words"]:
        bar = "#" * round(ws / 10) + "." * (10 - round(ws / 10))
        print(f"    {word:<15} [{bar}] {ws:3.0f}%  sounds like: {respell(phones)}")

    if r["issues"]:
        print("\n  What to fix:")
        for op, word, exp, got in r["issues"]:
            for msg, tip in explain(exp, got):
                print(f"    In '{word}': you {msg}")
                if tip:
                    print(f"      Tip: {tip}")
    print()
    return r


def verdict(score):
    return grade(score).verdict


def main():
    ap = argparse.ArgumentParser(description="Microphone pronunciation checker")
    ap.add_argument("text", nargs="?", help="sentence to practise (default: built-in list)")
    ap.add_argument("--file", help="assess a WAV file instead of the microphone")
    ap.add_argument("--device", type=int, help="input device index (see --list-devices)")
    ap.add_argument("--save", action="store_true", help="save each take as take_N.wav")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    proc, model = load_model()

    if args.file:
        text = args.text or PRACTICE_SENTENCES[0]
        assess(proc, model, load_wav(args.file), text)
        return

    sentences = [args.text] if args.text else PRACTICE_SENTENCES
    take = 0
    for sentence in sentences:
        while True:
            print("=" * 60)
            print(f'  Say: "{sentence}"')
            print("=" * 60)
            audio = record_from_mic(args.device)
            take += 1
            if args.save:
                sf.write(f"take_{take}.wav", audio, SAMPLE_RATE)
            assess(proc, model, audio, sentence)
            choice = input("[r]etry, [n]ext, [c]ustom sentence, [q]uit? ").strip().lower()
            if choice == "r":
                continue
            if choice == "c":
                sentence = input("Type your sentence: ").strip() or sentence
                continue
            if choice == "q":
                return
            break
    print("Done - nice practice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
