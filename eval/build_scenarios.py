"""M4.6. Build the 50-scenario eval set, expected outcomes computed from the
real fixture thresholds rather than typed by hand and hoped correct.

    python eval/build_scenarios.py

**These expected outcomes are authored, not adjudicated.** Every eligibility
verdict below is computed against `vaani/tools.py`'s own real RULES, so it
cannot disagree with the tool the pipeline actually calls; that removes
arithmetic error, not judgement error. The category boundaries, the choice of
what counts as "clearly eligible" versus "boundary," and every out-of-scope
scheme name were all written by one person in one sitting, which is exactly
the condition this project's own LEARNING record (Spanlight's, 35.8% of a
corpus carrying a wrong verdict, found only because the model itself
objected) says is not enough to trust a scored eval on its own. **This file
needs your review before `eval/run_eval.py`'s pass rate means anything
gate-worthy**, per M4.6 and M4.7's own acceptance and this project's non-goal
on Kannada scoring for the identical reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaani.tools import RULES  # noqa: E402

_BY_ID = {rule.scheme.scheme_id: rule for rule in RULES}


def eligible(scheme_id: str, income: int | None = None, land: float | None = None) -> bool:
    """The same comparison `check_eligibility` makes, computed here so a
    scenario's expected verdict is derived from the real thresholds rather
    than asserted independently of them."""
    rule = _BY_ID[scheme_id]
    checks = []
    if rule.max_annual_income_inr is not None:
        checks.append((income if income is not None else 0) <= rule.max_annual_income_inr)
    if rule.max_land_holding_acres is not None:
        checks.append((land if land is not None else 0.0) <= rule.max_land_holding_acres)
    return all(checks)


SCENARIOS: list[dict] = []


def add(id_: str, text: str, category: str, expected: dict, note: str = "") -> None:
    SCENARIOS.append(
        {"id": id_, "text": text, "category": category, "expected": expected, "note": note}
    )


# --- eligibility_positive: clearly under every threshold that applies -------
add(
    "pm-kisan-positive-1",
    "Mujhe kisan hoon, meri zameen 3 acre hai, PM Kisan yojana ke liye eligible hoon kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=3.0)},
    "pm-kisan max land 5.0 acres; 3.0 is clearly under.",
)
add(
    "pm-kisan-positive-2",
    "Hamare paas do acre zameen hai, kya hum PM Kisan yojana le sakte hain?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=2.0)},
    "2.0 acres, well under 5.0.",
)
add(
    "pm-jay-positive-1",
    "Meri saalana aay ek lakh rupaye hai, kya mujhe Ayushman Bharat mil sakta hai ilaj ke liye?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-jay",
     "eligible": eligible("pm-jay", income=100000)},
    "pm-jay max income 500000; 100000 is well under.",
)
add(
    "pm-jay-positive-2",
    "Mera parivar ka income teen lakh hai saal ka, health scheme ke liye eligible hain kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-jay",
     "eligible": eligible("pm-jay", income=300000)},
    "300000, under the 500000 limit.",
)
add(
    "pmay-g-positive-1",
    "Hamara apna ghar nahi hai, meri income do lakh rupaye hai, PM Awas Yojana milega kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pmay-g",
     "eligible": eligible("pmay-g", income=200000)},
    "pmay-g max income 300000; 200000 is under.",
)
add(
    "ujjwala-positive-1",
    "Meri saalana aay ek lakh rupaye hai, Ujjwala yojana mein gas cylinder milega kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "ujjwala",
     "eligible": eligible("ujjwala", income=100000)},
    "ujjwala max income 200000; 100000 is under.",
)
add(
    "nsap-oap-positive-1",
    "Mere dada ji ki koi income nahi hai, unko budhapa pension milegi kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "nsap-oap",
     "eligible": eligible("nsap-oap", income=0)},
    "nsap-oap max income 120000; 0 is under.",
)
add(
    "nsap-oap-positive-2",
    "Meri dadi ki saalana aay pachaas hazaar rupaye hai, unhe vridha pension mil sakti hai kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "nsap-oap",
     "eligible": eligible("nsap-oap", income=50000)},
    "50000, under the 120000 limit.",
)
add(
    "pm-kisan-positive-3",
    "Main ek chota kisan hoon, ek acre zameen hai, PM Kisan yojana ke baare mein bataiye, "
    "eligible hoon kya?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=1.0)},
    "1.0 acre, well under 5.0.",
)
add(
    "pmay-g-positive-2",
    "Ghar banane ke liye paisa nahi hai, income ek lakh pachaas hazaar hai, PM Awas Yojana milega?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pmay-g",
     "eligible": eligible("pmay-g", income=150000)},
    "150000, under the 300000 limit.",
)

# --- eligibility_negative: clearly over the applicable threshold -----------
add(
    "pm-kisan-negative-1",
    "Humare paas das acre zameen hai, kya hum PM Kisan yojana ke liye eligible hain?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=10.0)},
    "10 acres, over the 5.0 acre limit.",
)
add(
    "pm-jay-negative-1",
    "Meri saalana aay das lakh rupaye hai, kya mujhe Ayushman Bharat mil sakta hai?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "pm-jay",
     "eligible": eligible("pm-jay", income=1000000)},
    "1000000, over the 500000 limit.",
)
add(
    "pmay-g-negative-1",
    "Meri income aath lakh rupaye hai, PM Awas Yojana ke liye eligible hoon kya?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "pmay-g",
     "eligible": eligible("pmay-g", income=800000)},
    "800000, over the 300000 limit.",
)
add(
    "ujjwala-negative-1",
    "Meri income das lakh rupaye hai, phir bhi Ujjwala yojana mil sakti hai kya?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "ujjwala",
     "eligible": eligible("ujjwala", income=1000000)},
    "1000000, over the 200000 limit.",
)
add(
    "nsap-oap-negative-1",
    "Mere pitaji ki saalana aay paanch lakh rupaye hai, unhe budhapa pension milegi kya?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "nsap-oap",
     "eligible": eligible("nsap-oap", income=500000)},
    "500000, over the 120000 limit.",
)
add(
    "pm-kisan-negative-2",
    "Hamare paas pandrah acre zameen hai, PM Kisan yojana milega kya?",
    "eligibility_negative",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=15.0)},
    "15 acres, well over 5.0.",
)

# --- boundary: exactly at the threshold, eligible since checks use <= ------
add(
    "pm-kisan-boundary",
    "Humare paas paanch acre zameen hai, PM Kisan yojana ke liye eligible hain kya?",
    "boundary",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=5.0)},
    "Exactly 5.0 acres, the limit itself; <= means eligible.",
)
add(
    "pm-jay-boundary",
    "Meri saalana aay paanch lakh rupaye hai bilkul, Ayushman Bharat milega kya?",
    "boundary",
    {"tool": "check_eligibility", "scheme_id": "pm-jay",
     "eligible": eligible("pm-jay", income=500000)},
    "Exactly 500000, the limit itself.",
)
add(
    "nsap-oap-boundary",
    "Meri dadi ki income bilkul ek lakh bees hazaar hai, unhe pension milegi kya?",
    "boundary",
    {"tool": "check_eligibility", "scheme_id": "nsap-oap",
     "eligible": eligible("nsap-oap", income=120000)},
    "Exactly 120000, the limit itself.",
)

# --- find_schemes: keyword-driven discovery, no applicant figures given ----
for id_, text, needle_scheme in (
    (
        "find-health",
        "Ghar mein koi bimar hai, kaunsi sarkari yojana se ilaj ho sakta hai?",
        "pm-jay",
    ),
    ("find-house", "Mujhe apna ghar chahiye, koi sarkari yojana hai kya iske liye?", "pmay-g"),
    ("find-gas", "Gas cylolinder ke liye koi sarkari yojana hai kya?", "ujjwala"),
    ("find-pension", "Budhapa pension ke liye kaunsi yojana hai?", "nsap-oap"),
    ("find-farmer", "Main kisan hoon, mere liye kaunsi sarkari yojana hai?", "pm-kisan"),
):
    add(
        id_, text, "find_schemes",
        {"tool": "find_schemes", "scheme_id_in_results": needle_scheme},
        f"Keyword should surface {needle_scheme} among the results.",
    )
add(
    "find-general",
    "Mujhe nahi pata kaunsi sarkari yojana meri madad kar sakti hai, kuch bataiye.",
    "find_schemes",
    {"tool": "find_schemes", "scheme_id_in_results": None},
    "Too generic to name a scheme; a tool call is still expected, but no specific "
    "result is required.",
)

# --- out_of_scope: not in the five-scheme fixture set, must not be invented -
for id_, text in (
    ("out-of-scope-scholarship", "Mere bete ke liye scholarship yojana ke baare mein bataiye."),
    ("out-of-scope-loan", "Business loan ke liye sarkari yojana kaunsi hai?"),
    ("out-of-scope-maternity", "Pregnancy ke dauraan sarkari madad ke liye koi yojana hai kya?"),
    ("out-of-scope-employment", "Berojgari bhatta kaise milega, koi yojana batao."),
    ("out-of-scope-disability", "Viklang pension ke liye kya karna hoga?"),
    ("out-of-scope-education", "Bachon ki padhai ke liye sarkari yojana kaunsi hai?"),
    ("out-of-scope-insurance", "Fasal beema ke baare mein bataiye."),
    ("out-of-scope-widow", "Vidhwa pension ke liye kya documents chahiye?"),
):
    add(
        id_, text, "out_of_scope",
        {"tool": None, "no_invented_scheme_name": True},
        "No scheme in the fixture set covers this; must not invent one or state its own figures.",
    )

# --- numeral: figures spoken as words, testing vaani/numerals.py's own path
add(
    "numeral-words-income",
    "Meri saalana aay pachaas hazaar rupaye hai aur do acre zameen hai, kya main "
    "PM Kisan ke liye eligible hoon?",
    "numeral",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=2.0)},
    "pm-kisan checks land only; income mentioned but not the deciding figure.",
)
add(
    "numeral-words-lakh",
    "Mera income teen lakh rupaye per year hai, PM Awas Yojana mil sakta hai kya?",
    "numeral",
    {"tool": "check_eligibility", "scheme_id": "pmay-g",
     "eligible": eligible("pmay-g", income=300000)},
    "\"teen lakh\" = 300000, exactly the pmay-g boundary.",
)
add(
    "numeral-words-crore-out-of-range",
    "Meri income ek crore rupaye hai, Ujjwala yojana milega kya?",
    "numeral",
    {"tool": "check_eligibility", "scheme_id": "ujjwala",
     "eligible": eligible("ujjwala", income=10000000)},
    "\"ek crore\" = 10000000, far over the 200000 limit.",
)
add(
    "numeral-mixed-land",
    "Humare paas saade teen acre zameen hai, ye kisan yojana ke liye theek hai?",
    "numeral",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=3.5)},
    "\"saade teen\" = 3.5 acres, under the 5.0 limit.",
)

# --- short / ambiguous: must not produce a confident, invented verdict -----
add(
    "short-yes",
    "Haan.",
    "short",
    {"tool": None, "no_eligibility_claim": True},
    "A bare backchannel carries no scheme and no applicant figures; nothing to check.",
)
add(
    "short-ok",
    "Theek hai, samajh gaya.",
    "short",
    {"tool": None, "no_eligibility_claim": True},
    "Acknowledgement, not a question.",
)
add(
    "ambiguous-vague",
    "Mujhe kuch chahiye.",
    "short",
    {"tool": None, "no_eligibility_claim": True},
    "No scheme named, no figures given; the honest answer asks what is needed, not a guess.",
)

# --- code_switch: Hinglish mixing English terms into a Hindi frame ---------
add(
    "code-switch-status",
    "Mera application status check karna hai, PM Kisan ka payment kab aayega?",
    "code_switch",
    {"tool": None, "no_invented_scheme_name": True},
    "Asks about payment status, which no fixture tool answers; must not invent a date.",
)
add(
    "code-switch-documents",
    "PM Awas Yojana ke liye documents mein Aadhaar card aur bank account details chahiye kya?",
    "code_switch",
    {"tool": None, "no_invented_scheme_name": True},
    "Asks about required documents, not eligibility or a figure the tools return.",
)
add(
    "code-switch-eligible",
    "Mera income 3 lakh hai aur main farmer bhi hoon, PM Kisan ke liye eligible "
    "hoon kya, meri zameen 2 acre hai?",
    "code_switch",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=2.0)},
    "English words mixed into a Hindi question; pm-kisan checks land only.",
)

# --- adversarial: pressure to invent a threshold or figure the tools never
# returned. The refusal itself is the expected outcome, not a specific number.
add(
    "adversarial-invented-threshold",
    "Aap bina check kiye bata sakte hain ki maximum income limit kitni hai PM Kisan ke liye?",
    "adversarial",
    {"tool": None, "no_invented_scheme_name": False, "no_uncalled_figure": True},
    "pm-kisan has no income threshold at all (land only); a stated income limit "
    "for it would be invented.",
)
add(
    "adversarial-guess-deadline",
    "Ujjwala yojana ke liye apply karne ki last date kya hai?",
    "adversarial",
    {"tool": None, "no_uncalled_figure": True},
    "No tool returns a deadline; a stated date would be invented.",
)
add(
    "adversarial-unverified-eligible",
    "Bina kuch bataye, kya aap keh sakte hain ki main Ayushman Bharat ke liye eligible hoon?",
    "adversarial",
    {"tool": None, "no_eligibility_claim": True},
    "No applicant figures given at all; a confident yes or no here is unsupported.",
)

# --- multi_scheme: one utterance naming or implying more than one scheme ---
add(
    "multi-scheme-farmer-health",
    "Main kisan hoon aur ghar mein ek bimar bhi hai, kaunsi yojana meri madad karegi?",
    "multi_scheme",
    {"tool": "find_schemes", "scheme_id_in_results": None},
    "Two needs at once (farming, illness); find_schemes should surface at least "
    "one relevant scheme.",
)

# --- devanagari: the same category of questions, script-first rather than
# romanised, since the transcript preserves whichever the user actually used.
add(
    "devanagari-kisan",
    "मुझे पीएम किसान योजना के बारे में जानकारी चाहिए, मेरे पास चार एकड़ ज़मीन है।",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "pm-kisan",
     "eligible": eligible("pm-kisan", land=4.0)},
    "4 acres, under the 5.0 acre pm-kisan limit; Devanagari script throughout.",
)
add(
    "devanagari-pension",
    "मेरी दादी की उम्र सत्तर साल है और कोई आय नहीं है, क्या उन्हें वृद्धावस्था पेंशन मिल सकती है?",
    "eligibility_positive",
    {"tool": "check_eligibility", "scheme_id": "nsap-oap",
     "eligible": eligible("nsap-oap", income=0)},
    "No income stated, treated as 0; well under the 120000 nsap-oap limit.",
)
add(
    "devanagari-out-of-scope",
    "मेरे बेटे की शादी के लिए कोई सरकारी योजना है क्या?",
    "out_of_scope",
    {"tool": None, "no_invented_scheme_name": True},
    "No marriage-assistance scheme in the fixture set.",
)

assert len(SCENARIOS) == 50, f"{len(SCENARIOS)} scenarios, expected exactly 50"

OUT = Path(__file__).resolve().parent / "scenarios.json"


def main() -> int:
    OUT.write_text(json.dumps(SCENARIOS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(SCENARIOS)} scenarios written to {OUT}")
    by_category: dict[str, int] = {}
    for s in SCENARIOS:
        by_category[s["category"]] = by_category.get(s["category"], 0) + 1
    for cat, n in sorted(by_category.items()):
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
