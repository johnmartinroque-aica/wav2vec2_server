"""Kid-friendly coaching text from a local Ollama model.

The rule-based issues from phonemes.explain() are passed in as facts, so the
model only rewrites them into simple advice; it never judges the audio itself.
If Ollama is not running, coach() returns None and the API still works.
"""
import json
import os
import re
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))
MAX_ISSUES = 2  # a young child can only work on a couple of sounds at once

SYSTEM_PROMPT = """You are a kind, cheerful English pronunciation coach for a young Japanese child (about 6-10 years old).

Rules:
- Use very short, simple English sentences. Easy words only. No phonetic symbols.
- Talk directly to the child as "you". Never say "the child" and never repeat these notes back.
- Start with one line of warm praise.
- Then talk ONLY about the problems listed. Never invent other problems.
- Never say "wrong", "not right", "bad" or "incorrect". Say "let's try" or "almost" instead.
- For each problem, say the word and the sound, then explain the tip in easy words.
  Keep the tip's meaning exactly: do not change what the tongue, lips, or teeth do.
- End with one fun practice idea (for example: say it slowly 3 times, or look in a mirror).
- At most 5 short sentences in total. Use one or two friendly emoji."""

# Extra vowels after consonants ("dogu") are the most common Japanese-accent habit.
EXTRA_VOWEL_TIP = ('Japanese words end in a vowel, but English words often do not. '
                   'Stop the word right after the last sound, with no "u" or "o" at the end.')


def _prompt(text, score, issues):
    if not issues:
        return (f'The child said "{text}" and got {score}/100. Every sound was correct! '
                "Give 2 short sentences of happy praise. Do not ask them to try again.")
    lines = []
    for i in issues:
        line = f'- In the word "{i["word"]}", the child {i["message"].removeprefix("you ")}.'
        tip = i.get("tip") or (EXTRA_VOWEL_TIP if "added an extra" in i["message"] else None)
        if tip:
            line += f" Tip: {tip}"
        lines.append(line)
    return (f'The child tried to say: "{text}"\nScore: {score}/100\n'
            "Problems found:\n" + "\n".join(lines) +
            "\n\nWrite your friendly advice for the child now.")


def coach(text, score, issues):
    """Return advice text for the child, or None if Ollama is unavailable."""
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _prompt(text, score, issues[:MAX_ISSUES])}],
        "options": {"temperature": 0.4, "num_predict": 250},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            text = json.loads(resp.read())["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama feedback unavailable: {e}")
        return None
    return _drop_cut_off_sentence(_drop_harsh_sentences(text)) or None


# The 1B model doesn't always follow the "never say wrong" rule, so filter it.
_HARSH = re.compile(r"\b(wrong|not right|incorrect|bad|mistakes?)\b", re.IGNORECASE)


def _drop_harsh_sentences(text):
    # Each match is one sentence plus the whitespace after it, so line breaks survive.
    sentences = re.findall(r"[^.!?]*(?:[.!?]+|$)\s*", text)
    return "".join(x for x in sentences if not _HARSH.search(x)).strip()


def _drop_cut_off_sentence(text):
    """If the reply hit the length limit mid-sentence, end it at the last full sentence."""
    if text.endswith((".", "!", "?", ")", '"')) or not any(c in text for c in ".!?"):
        return text
    return text[:max(text.rfind(c) for c in ".!?") + 1]
