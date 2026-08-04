# Contributing

Conventions for this repo and every project forked from the same chassis.

Single developer, single assistant. There is no second agent and no parallel
work stream, so nothing here assumes one: no delegation tags in backlogs, no
branch-per-agent rules, and time estimates assume one worker.

## Stack

- Python 3.13, `uv` for dependency management (`[tool.uv] package = false`,
  this is an application, not a published package).
- `app/` is the FastAPI service (factory in `app/main.py`).
- `llm/` is the provider-agnostic chat client. Changes here should be
  conservative and well tested.
- `docker compose up` runs the app plus local postgres (pgvector) and redis.
- CI runs ruff and pytest on every PR.

## Working conventions

1. **Plan before code.** For anything beyond a trivial fix, propose the approach
   and get sign-off before writing files.
2. **Tasks come only from BACKLOG.md.** No ad-hoc scope mid-session. Add it to
   the backlog first, or ask whether it belongs there.
3. **Every nontrivial choice gets a DECISIONS.md entry**, written when the
   choice is made rather than reconstructed later.
4. **Small diffs.** Several focused changes beat one large one.
5. **Tests for every acceptance criterion.** If a task has a defined "done",
   there is a test proving it.
6. **Never touch files outside the current task's scope.** File drive-by
   refactors as backlog items instead.
7. **Update BACKLOG.md checkboxes at the end of each session.**

## Commit granularity

Batch a working session into a single commit, two at most. Commit straight to
the default branch. These are solo projects, so a branch-and-pull-request cycle
for routine work adds ceremony without adding review value.

The exception is a pull request that is itself a deliverable, such as the
regression-blocked PR that ShipGate publishes as a proof artifact. Those exist
because the artifact needs them, not as a workflow habit.

Never pad the history. A commit covering a session should still read as one
coherent change with a message that explains it.

## Writing style

This is a portfolio repo. It should read as though one engineer wrote it,
because one engineer is responsible for it. Anything that reads as machine
generated undermines that.

Applies to README.md, SPEC.md, BACKLOG.md, DECISIONS.md, code comments, commit
messages, and PR descriptions.

**Never use:**

- Em dashes or en dashes. Use a comma, a colon, parentheses, or two sentences.
- Emoji in headings, tables, or status banners.
- Filler adjectives: comprehensive, robust, seamless, powerful, cutting-edge,
  production-grade (unless literally measured), blazing fast.
- "Leverage" as a verb. Use "use".
- The "it's not just X, it's Y" construction.
- "Let's dive in", "in today's landscape", "at its core", "the key insight is".
- Bold lead-ins on every bullet in a list. Vary the shape.
- Perfectly parallel three-item lists where two items would do.

**Do:**

- Write plainly and specifically. Name real numbers, real file paths, real
  failures.
- Vary sentence length. Some short.
- Let the "What Broke" section be genuinely unflattering. That section is the
  most credible thing in the README.
- Prefer concrete verbs over abstractions.

**Commit messages:** imperative mood, no co-author trailers, no tool
attribution, no generated-by footers.

## Secrets

Never commit credentials. This repo is public. Connection strings and API keys
live in `.env` locally (gitignored), in GitHub Actions secrets for CI, and in
Render environment variables for the deployed service.
