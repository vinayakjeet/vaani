"""Synthesise the fixed audio corpus once, so every measured run drives the same input.

    python scripts/build_corpus.py

M4.2, and SPEC A8's requirement stated plainly: the input must not vary between
configurations, or a difference in the numbers could be a difference in what was said
rather than a difference in how fast it was answered.

**This corpus is synthesised, not recorded, and that is a real methodological limit
stated here rather than discovered later.** Nobody with a microphone was available to
record twenty utterances by hand. What is checked in instead is deterministic,
byte-identical on every rebuild, and covers the same scenario classes a recorded corpus
would: the five known schemes, an out-of-scope question, short and ambiguous utterances,
and numbers spoken the way a caller actually says them. What it does not cover is
anything a recorded corpus is honest about being good at: real disfluency, background
noise, accent variation, a microphone held at the wrong distance. `bench/waterfall.py`'s
own numbers are latency, not recognition accuracy, and latency is what a synthesised
corpus can measure faithfully; whichever report reads this corpus's numbers has to say
so, not just this file.

Text-to-speech twice over is also not free of its own circularity: this corpus is
synthesised by the same `EdgeTts` class the pipeline itself uses for its answers, so a
systematic bias in Microsoft's Hindi voice does not show up as a surprise in either
direction. Recorded human speech is the fix, and it is M4.2's own next step once someone
is available to do it.
"""

from __future__ import annotations

import asyncio
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import miniaudio  # noqa: E402

from vaani.protocol import SAMPLE_RATE  # noqa: E402
from vaani.tts import VOICE_HI, EdgeTts  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / "bench" / "corpus"
MANIFEST = CORPUS / "manifest.json"


@dataclass(frozen=True)
class Utterance:
    id: str
    text: str
    category: str


# Twenty, per M4.2's acceptance. Categories name what each one is for, not a script
# direction: `scheme` utterances exercise `check_eligibility`/`find_schemes` for all
# five known schemes; `out_of_scope` exercises the refusal path a scheme genuinely
# outside the fixture set has to take; `numeral` exercises M1.10's normalisation, income
# and land spoken the way a caller actually says them rather than as clean digits;
# `short` is at or below the length `TurnTaking` treats as a backchannel, so the corpus
# itself can be used to check that boundary later without inventing new audio for it;
# `code_switch` mixes English terms into a Hindi sentence, which is the "Hinglish" half
# of the project's own name.
UTTERANCES: tuple[Utterance, ...] = (
    Utterance(
        "scheme-pm-kisan", "Mujhe kisan hoon, PM Kisan yojana ke baare mein bataiye.", "scheme"
    ),
    Utterance(
        "scheme-pm-jay",
        "Mera ilaj ke liye paisa nahi hai, Ayushman Bharat se madad mil sakti hai kya?",
        "scheme",
    ),
    Utterance(
        "scheme-pmay-g",
        "Hamara apna ghar nahi hai, PM Awas Yojana ke liye eligible hoon kya?",
        "scheme",
    ),
    Utterance(
        "scheme-ujjwala", "Ujjwala yojana mein gas cylinder kaise milega, mujhe batao.", "scheme"
    ),
    Utterance(
        "scheme-nsap-oap",
        "Mere dada ji budhapa pension ke liye apply karna chahte hain, kya karna hoga?",
        "scheme",
    ),
    Utterance(
        "eligibility-income-words",
        "Meri saalana aay pachaas hazar rupaye hai aur do acre zameen hai, "
        "kya main PM Kisan ke liye eligible hoon?",
        "numeral",
    ),
    Utterance(
        "eligibility-income-digits",
        "Mera income 3 lakh rupaye per year hai, PM Awas Yojana mil sakta hai kya?",
        "numeral",
    ),
    Utterance(
        "eligibility-land",
        "Humare paas saade teen acre zameen hai, ye kisan yojana ke liye theek hai?",
        "numeral",
    ),
    Utterance(
        "find-schemes-health",
        "Ghar mein koi bimar hai, kaunsi sarkari yojana se ilaj ho sakta hai?",
        "scheme",
    ),
    Utterance(
        "find-schemes-general",
        "Mujhe nahi pata kaunsi sarkari yojana meri madad kar sakti hai, kuch bataiye.",
        "scheme",
    ),
    Utterance(
        "out-of-scope-scholarship",
        "Mere bete ke liye scholarship yojana ke baare mein bataiye.",
        "out_of_scope",
    ),
    Utterance(
        "out-of-scope-loan", "Business loan ke liye sarkari yojana kaunsi hai?", "out_of_scope"
    ),
    Utterance("short-yes", "Haan.", "short"),
    Utterance("short-ok", "Theek hai, samajh gaya.", "short"),
    Utterance(
        "code-switch-status",
        "Mera application status check karna hai, PM Kisan ka payment kab aayega?",
        "code_switch",
    ),
    Utterance(
        "code-switch-documents",
        "Documents mein Aadhaar card aur bank account details chahiye kya?",
        "code_switch",
    ),
    Utterance(
        "code-switch-online",
        "Online form fill kar sakte hain ya office jaana padega registration ke liye?",
        "code_switch",
    ),
    Utterance(
        "devanagari-kisan",
        "मुझे पीएम किसान योजना के बारे में जानकारी चाहिए, मैं किसान हूं।",
        "scheme",
    ),
    Utterance(
        "devanagari-pension",
        "मेरी दादी की उम्र सत्तर साल है, क्या उन्हें वृद्धावस्था पेंशन मिल सकती है?",
        "scheme",
    ),
    Utterance(
        "negative-not-eligible",
        "Meri income das lakh rupaye hai, phir bhi Ujjwala yojana mil sakti hai kya?",
        "numeral",
    ),
)


async def build() -> int:
    CORPUS.mkdir(parents=True, exist_ok=True)
    tts = EdgeTts()
    manifest: list[dict] = []
    written = 0

    for utterance in UTTERANCES:
        path = CORPUS / f"{utterance.id}.wav"
        mp3 = bytearray()
        async for chunk in tts.synthesize(utterance.text, VOICE_HI):
            mp3 += chunk

        decoded = miniaudio.decode(
            bytes(mp3),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=SAMPLE_RATE,
        )
        pcm = decoded.samples.tobytes()

        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)

        duration_s = len(pcm) / 2 / SAMPLE_RATE
        manifest.append(
            {
                "id": utterance.id,
                "text": utterance.text,
                "category": utterance.category,
                "duration_s": round(duration_s, 3),
                "file": path.name,
            }
        )
        written += 1
        print(f"  {utterance.id}: {duration_s:.2f}s")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return written


if __name__ == "__main__":
    count = asyncio.run(build())
    print(f"{count} utterances written to {CORPUS}")
