# Vaani's Hindi/Hinglish eligibility eval set

50 scripted scenarios exercising a voice agent that answers Indian government
welfare-scheme eligibility questions over a small, fixed fixture of five
schemes. Built and scored against a live pipeline, not a mock: every
scenario is a real question sent through the actual language model and
tool-calling code this project deploys, and `expected` is what the correct
tool call and verdict should have been, not what the model happened to say.

## Methodology

`build_scenarios.py` writes `scenarios.json`. Every eligibility verdict in it
is computed programmatically from the same threshold data the pipeline's own
`check_eligibility` tool checks against (`vaani/tools.py`'s `RULES`), via a
small `eligible(scheme_id, income, land)` helper that mirrors that tool's own
comparison logic, rather than typed by hand and hoped correct. That removes
arithmetic error from the labels. It does not remove judgement error: which
scenario belongs in which category, what counts as "clearly eligible" versus
"boundary," and every out-of-scope scheme name were all written by one
person in one sitting, and this file's own header says plainly that this
needs an independent review pass before a pass rate against it should gate
anything. Publishing a labelled eval set as though its labels were beyond
question, when nobody but the author has actually checked them, is a
specific, named mistake this project is trying not to repeat; see `run_eval.py`'s
own header for the fuller account of why.

`run_eval.py` runs each scenario through the real pipeline (`vaani.llm_turn.StreamedTurn`),
recording every tool call the model actually made, successful or rejected,
and checks the recording against each scenario's `expected` dict: which
tool, which scheme id, which eligibility verdict, whether a `find_schemes`
result contained the right scheme, whether a reply avoided claiming an
eligibility verdict or inventing a figure no tool ever returned. It does not
grade the reply's prose. Matching on generated text would be either far too
brittle, since the same correct answer can be phrased many ways, or would
need its own natural-language layer whose own correctness this eval would
then be trusting blind. The structured facts the pipeline actually decided
are what gets checked.

## Categories

| Category | Count | Tests |
|---|---|---|
| `eligibility_positive` | 12 | Applicant clearly under every threshold that applies |
| `eligibility_negative` | 6 | Applicant clearly over a threshold |
| `boundary` | 3 | Applicant exactly at a threshold, where `<=` decides it |
| `find_schemes` | 6 | Keyword search returns the right scheme among results |
| `out_of_scope` | 9 | No scheme in the five-scheme fixture covers the question |
| `numeral` | 4 | Figures spoken as words ("teen lakh," "saade teen") rather than digits |
| `short` | 3 | Backchannel-like utterances carrying no question at all |
| `code_switch` | 3 | English terms mixed into a Hindi frame |
| `adversarial` | 3 | Direct pressure to state a figure, deadline, or verdict no tool provided |
| `multi_scheme` | 1 | One utterance naming or implying more than one scheme |

Three of the fifty scenarios are written in Devanagari script rather than
romanised Hindi, to check the same categories hold when the transcript
arrives in the script the user actually spoke in rather than transliterated.

## Kannada is not in this eval set, and that is a stated limitation, not an oversight

The deployed agent's language handling is not specific to Hindi and
Hinglish; nothing in the pipeline itself is Hindi-only. This eval set is,
because the people who authored and reviewed its labels can personally read
and adjudicate Hindi and Hinglish, and cannot do the same for Kannada. An
eval set's expected outcomes are only as trustworthy as the review behind
them, and a scored "correct" verdict that nobody involved can actually
verify is worse than a smaller claim that says plainly what it does and does
not cover. Extending this eval set to another language requires someone who
can genuinely adjudicate that language's scenarios, not a machine
translation of these fifty.

## Reproducing a result

```
python eval/build_scenarios.py   # regenerates scenarios.json from vaani/tools.py's own rules
python eval/run_eval.py          # runs all 50 against the live pipeline; needs GROQ_API_KEY
```

`results/` carries the raw output of one such run, committed so a reader can
inspect exactly which scenarios passed or failed without spending a live API
call themselves; see `results/README.md`. A fresh run will not reproduce it
exactly, since the underlying model's real tool-calling behaviour varies run
to run, which is itself part of what a small, fixed eval set like this one
is for: catching a regression large enough to matter against noise this
size, not claiming zero noise exists.
