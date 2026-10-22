"""Side-by-side comparison of two wav2vec2 phoneme models on the same recording.

Run:  python compare.py   then open http://127.0.0.1:7860 in your browser.

Each model is downloaded once (cached by Hugging Face in ~/.cache/huggingface)
and loaded into memory once at startup.
"""
import html
import time
from difflib import SequenceMatcher

import gradio as gr
import numpy as np

from phonemes import explain, ipa, respell
from pronounce import PRACTICE_SENTENCES, SAMPLE_RATE, analyze
from scoring import GRADES, grade
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_IDS = [
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
]

# Short background shown above each model's results.
MODEL_INFO = {
    "facebook/wav2vec2-lv-60-espeak-cv-ft":
        "Learned by listening to thousands of hours of English audiobooks. "
        "It knows native English well, but may overlook small mistakes.",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft":
        "Learned by listening to people speaking 53 different languages. "
        "It may be better at noticing accents and mistakes made by learners.",
}


def header(model_id):
    return (f"<h3>{html.escape(model_id)}</h3>"
            f"<p class='pc-info'>{html.escape(MODEL_INFO[model_id])}</p>")


def load(model_id):
    print(f"Loading {model_id} (first run downloads ~1.2 GB)...")
    proc = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    return proc, model


MODELS = {mid: load(mid) for mid in MODEL_IDS}


def to_mono_16k(audio):
    """Gradio gives (sample_rate, int16/float array); convert to float32 mono 16 kHz."""
    sr, data = audio
    data = np.asarray(data)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / sr)
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data)
    return data.astype(np.float32)


def short(model_id):
    """'facebook/wav2vec2-lv-60-espeak-cv-ft' -> 'lv-60'"""
    return "-".join(model_id.split("/")[-1].split("-")[1:3])


def word_timing(r):
    """Per-word forced-alignment info, or Nones if alignment failed."""
    return r["timing"] or [None] * len(r["words"])


def render(model_id, r, secs):
    if r is None:
        return f"<div class='pc-result'>{header(model_id)}<p class='pc-msg'>Recording too short or too quiet.</p></div>"
    g = grade(r["score"])
    chips = []
    for (w, ph, ws), t in zip(r["words"], word_timing(r)):
        tip = "No timing (alignment failed)"
        if t and t["start"] is not None:
            tip = f"Said at {t['start']:.2f}-{t['end']:.2f}s"
        chips.append(
            f'<span class="pc-word" title="{html.escape(tip)}" '
            f'style="border-color:{grade(ws).color};color:{grade(ws).color}">'
            f'{html.escape(w)}<em>{html.escape(respell(ph))}</em><small>{ws:.0f}%</small></span>')
    words = "".join(chips)
    rows = [
        f"<li>In <b>{html.escape(word)}</b>: you {html.escape(msg)}</li>"
        for op, word, exp, got in r["issues"]
        for msg, _ in explain(exp, got)
    ]
    details = (f"<h4>Issues</h4><ul>{''.join(rows)}</ul>" if rows
               else "<p>No mistakes detected.</p>")
    n_exp, n_heard = len(r["expected"]), len(r["spoken"])
    matched = sum(b.size for b in SequenceMatcher(
        None, r["expected"], r["spoken"], autojunk=False).get_matching_blocks())
    math = (f"<p class='pc-math'>{matched} sounds matched · {n_exp} expected · {n_heard} heard<br>"
            f"Score = 2 × {matched} ÷ ({n_exp} + {n_heard}) = {r['score']:.0f}</p>")
    expected_ipa = "  ".join(ipa(ph) for _, ph, _ in r["words"])
    heard_ipa = "  ".join(ipa(g) if g else "(none)" for g in r["heard_words"])
    return f"""
<div class="pc-result">
  {header(model_id)}
  <div class="pc-score" style="color:{g.color}">{r["score"]:.0f}<span>/100</span></div>
  <div class="pc-verdict" style="color:{g.color}">Grade {g.letter} - {g.verdict}</div>
  <div class="pc-time">{secs:.2f}s</div>
  {math}
  <div class="pc-words">{words}</div>
  <table class="pc-phones">
    <tr><th>Should sound like (IPA)</th><td>{html.escape(expected_ipa)}</td></tr>
    <tr><th>Heard (IPA)</th><td>{html.escape(heard_ipa)}</td></tr>
  </table>
  {details}
</div>"""


def summary(results):
    """Per-word score table for both models, plus the overall difference."""
    (a_id, a, _), (b_id, b, _) = results
    if a is None or b is None:
        return ""
    na, nb = html.escape(short(a_id)), html.escape(short(b_id))
    head = f"<tr><th>Word</th><th>{na}</th><th>{nb}</th><th>Diff</th></tr>"
    body = "".join(
        f"<tr><td>{html.escape(w)}</td><td>{sa:.0f}%</td><td>{sb:.0f}%</td><td>{sb - sa:+.0f}</td></tr>"
        for (w, _, sa), (_, _, sb) in zip(a["words"], b["words"])
    )
    total = (f"<tr><th>Overall</th><th>{a['score']:.0f}</th><th>{b['score']:.0f}</th>"
             f"<th>{b['score'] - a['score']:+.0f}</th></tr>")
    return f"<div class='pc-result'><h3>Comparison</h3><table class='pc-cmp'>{head}{body}{total}</table></div>"


def check(audio, sentence):
    sentence = (sentence or "").strip()
    if not sentence:
        msg = "<p class='pc-msg'>Type or pick a sentence first.</p>"
        return msg, "", ""
    if audio is None:
        msg = "<p class='pc-msg'>Record yourself first (click the microphone).</p>"
        return msg, "", ""
    data = to_mono_16k(audio)
    results = []
    for mid, (proc, model) in MODELS.items():
        t0 = time.perf_counter()
        r = analyze(proc, model, data, sentence)
        results.append((mid, r, time.perf_counter() - t0))
    return (render(*results[0]), render(*results[1]), summary(results))


CSS = """
.pc-result { text-align:center; padding:8px; }
.pc-result h3 { font-size:15px; opacity:.7; margin-bottom:8px; }
.pc-info { font-size:13px; opacity:.75; max-width:460px; margin:0 auto 12px; }
.pc-score { font-size:56px; font-weight:700; line-height:1; }
.pc-score span { font-size:22px; opacity:.6; }
.pc-verdict { font-size:18px; font-weight:600; margin:4px 0; }
.pc-time { font-size:13px; opacity:.6; margin-bottom:8px; }
.pc-math { font-size:13px; opacity:.75; margin:0 0 12px; }
.pc-words { display:flex; flex-wrap:wrap; gap:6px; justify-content:center; margin-bottom:12px; }
.pc-word { border:2px solid; border-radius:10px; padding:4px 10px; font-size:17px; font-weight:600;
           display:inline-flex; flex-direction:column; align-items:center; }
.pc-word em { font-size:12px; font-style:normal; font-weight:400; opacity:.85; }
.pc-word small { font-size:11px; font-weight:400; }
.pc-phones { margin:0 auto 12px; font-size:14px; border:none !important; }
.pc-phones th, .pc-phones td { border:none !important; padding:3px 6px; }
.pc-phones th { text-align:right; opacity:.7; font-weight:400; }
.pc-phones td { text-align:left; font-weight:600; word-spacing:6px; }.pc-result ul { text-align:left; display:inline-block; }
.pc-cmp { margin:0 auto; }
.pc-cmp td, .pc-cmp th { padding:4px 12px; text-align:center; }
.pc-msg { text-align:center; padding:24px; opacity:.8; }
"""

with gr.Blocks(title="Model Comparison") as demo:
    gr.Markdown("# ⚖️ Model Comparison\nRecord once - both models score the same audio.")
    with gr.Accordion("📋 Instructions for testers", open=True):
        gr.Markdown(
            "1. **Pick a sentence** from the list (or type your own).\n"
            "2. **Click the microphone** 🎙️, then **Record** "
            "(allow microphone access if your browser asks).\n"
            "3. **Read the sentence aloud** at a normal speed.\n"
            "4. **Click stop** ⏹️ - both models score the same recording automatically "
            "(or press **Compare**).\n"
            "5. **Compare the results**: each model's column shows its score, per-word colors "
            "and the sounds it heard; the table at the bottom shows per-word differences.\n"
            "6. **To try again**, click **✖** on the recording to clear it, then record again.\n\n"
            "**What to test:**\n"
            "- Say a sentence **correctly** - a good model should give a high score.\n"
            "- Say it **with a deliberate mistake** (e.g. *ship* → *sip*, *rock* → *lock*) - "
            "a good model should lower the score for that word.\n"
            "- Try **different speakers and accents**, and note which model's *Heard* line "
            "matches what was actually said.\n"
            "- Note the **time** under each score - lower is faster.\n\n"
            "**Tips:** use a quiet room, keep the mic close, and wait a moment before speaking. "
            "You can also upload a recording instead of using the mic."
        )
    with gr.Accordion("🧮 How the score and grade are calculated", open=False):
        gr.Markdown(
            "**1. Expected sounds.** The sentence is broken into the sounds a native "
            "speaker would make. *\"Rock the\"* becomes **r-ah-k th-uh** (5 sounds).\n\n"
            "**2. Heard sounds.** The model listens to the recording and writes down the "
            "sounds it heard, e.g. **l-ah-k z-uh** (5 sounds).\n\n"
            "**3. Matching.** The two lists are lined up and the sounds that match, "
            "in the same order, are counted. Here *ah*, *k* and *uh* match = **3**.\n\n"
            "**4. Overall score.**\n\n"
            "> Score = 2 × matched ÷ (expected + heard) × 100\n"
            "> = 2 × 3 ÷ (5 + 5) × 100 = **60**\n\n"
            "Wrong sounds lower the score, and so do **missing** sounds "
            "and **extra** sounds (like the extra vowel in *milk* → *mi-ru-ku*), "
            "because extra sounds make the heard list longer.\n\n"
            "**5. Word score.** The share of that word's expected sounds that matched. "
            "*rock* (r-ah-k) with *ah* and *k* matched = 2 of 3 = **67%**. "
            "Extra sounds don't lower a word score - only the overall score.\n\n"
            "**6. Grade.** The overall score decides the grade:\n\n"
            "| Grade | Score | Meaning |\n|---|---|---|\n"
            + "".join(
                f"| **{g.letter}** - {g.verdict} | "
                f"{g.min}-{GRADES[i - 1].min - 1 if i else 100} | {g.meaning} |\n"
                for i, g in enumerate(GRADES)
            )
            + "\nWord colors use the same bands: green / light green = A / B, "
            "orange = C, red = D.\n\n"
            "**7. Timing.** The app also finds *when* each word was said in the recording "
            "(hover over a word to see it). This is used to split the *Heard* line into words, "
            "so an extra vowel stays with the word it belongs to (*milk* → *mi-ru-ku*).\n\n"
            "Each model's result also shows its own numbers, e.g. "
            "*3 sounds matched · 5 expected · 5 heard*, so you can check how the score came about."
        )
    with gr.Row():
        sentence = gr.Dropdown(PRACTICE_SENTENCES, value=PRACTICE_SENTENCES[0],
                               allow_custom_value=True, label="Sentence (pick or type your own)")
        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
    btn = gr.Button("Compare", variant="primary")
    with gr.Row():
        out_a = gr.HTML(f"<div class='pc-result'>{header(MODEL_IDS[0])}</div>")
        out_b = gr.HTML(f"<div class='pc-result'>{header(MODEL_IDS[1])}</div>")
    out_cmp = gr.HTML()

    btn.click(check, [audio, sentence], [out_a, out_b, out_cmp])
    audio.stop_recording(check, [audio, sentence], [out_a, out_b, out_cmp])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CSS, share=True)
