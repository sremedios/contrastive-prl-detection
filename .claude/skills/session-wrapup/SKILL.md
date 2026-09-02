---
name: session-wrapup
description: >-
  End-of-session wrap-up routine for a coding project. Writes a
  resumption-friendly entry to a rolling DEVLOG.md at the repo root, verifies the
  project version is consistent across every declaration site (pyproject.toml,
  __init__.py, setup.cfg, VERSION, etc.), then stages, commits with a Conventional
  Commits message, and pushes. Use this whenever the user signals they're closing
  out a working session — phrasings like "wrap up", "let's close out", "I'm done
  for today", "write notes/documentation so we can pick up later", "document this
  session", "log where we left off", or any request that bundles "write session
  notes + make the version consistent + commit and push". Trigger it even when the
  user mentions only part of the routine (just "commit and push as we wrap up", or
  just "jot down where we left off"), because at session end the full routine is
  almost always what they want.
---

# Session wrap-up

A fixed routine for cleanly ending a working session on a code project. It exists
because the same three chores happen every time work stops, and doing them
consistently is what makes the *next* session — whether picked up by a human or a
fresh Claude instance with no memory of today — start fast instead of cold.

Run the steps in this order: **(1) sync the version, (2) write the DEVLOG entry,
(3) commit and push.** Version first because any fix you make is a code change the
DEVLOG should record; the DEVLOG before the commit so it ships in the same commit.

If you're not in a git repository, say so and stop — the routine assumes one.

---

## Step 1 — Make the version consistent

The goal is narrow and important: every place the project declares its version
should say the **same** thing. This is a *consistency* check, **not** a bump — do
not increment the version unless the user explicitly asks.

Run the bundled scanner from the repo root:

```bash
python <skill_dir>/scripts/check_versions.py .
```

It prints every version declaration it finds (file, what kind, the value) and a
verdict. Read the verdict:

- **All agree** → nothing to do; note the version, move on.
- **Mismatch** → pick the source of truth and propagate it to the others with
  ordinary file edits (so the changes are visible and reviewable). Default source
  of truth is `pyproject.toml`'s `[project] version`; if that's absent, use the
  package's `__version__`. If it's genuinely unclear which value is intended,
  ask the user rather than guessing — picking the wrong one silently is worse
  than a five-second question.
- **Dynamic/VCS versioning detected** (setuptools-scm, hatch-vcs, versioneer, or
  `dynamic = ["version"]`) → the version is computed from git tags, so there's
  nothing to hand-sync. Don't "fix" hard-coded fallbacks to match — that's the
  intended setup, not a bug. Just report it.

The scanner only reports; you do the edits. That keeps every change to the
version in the normal edit stream where the user can see it.

---

## Step 2 — Write the DEVLOG entry

Append a new entry to **`DEVLOG.md`** at the repo root, **newest on top**, so the
most recent "here's where we are" is the first thing anyone sees on opening the
file. If `DEVLOG.md` doesn't exist, create it with a one-line title header first.

### What this entry is for

Write it as a **handoff to a competent colleague who has zero memory of today's
session but is expected to continue the work tomorrow.** That colleague might be
the user, might be a future you. The test of a good entry: could someone pick it
up and resume without re-reading the whole diff or re-deriving the decisions you
already made? Optimize for *resumption*, not for a changelog.

That framing matters more than the template. The template is a checklist so you
don't forget a dimension; it is not a form to pad. **Omit any section that would
be empty** rather than writing "N/A" — empty scaffolding is noise that makes the
useful parts harder to find.

### Template

```markdown
## YYYY-MM-DD — <short session title>
**Branch:** <branch> · **Version:** <version>

### Summary
One to three sentences: what this session set out to do and what changed.

### Changes
- The substantive changes, in plain language, grouped logically.
  Describe intent and effect — not a restatement of the git diff.

### Decisions & rationale
- <decision> — <why>. Capture the non-obvious tradeoffs and the roads not taken,
  so future-you doesn't relitigate a question that's already settled.

### State
- Working / verified: ...
- In progress / partial: ...
- Known issues: ...

### Next steps
- [ ] Specific, actionable items — name the files, functions, or commands.
- Call out the single most important thing to do next.

### Notes & gotchas
- Non-obvious context a fresh session would otherwise have to rediscover:
  required env vars, where data lives, a load-bearing hack not to touch, a
  flaky test, an external dependency's quirk.
```

Fill `<branch>` from `git rev-parse --abbrev-ref HEAD` and `<version>` from the
value confirmed in Step 1.

### Writing guidance

- **Next steps and gotchas are the highest-value sections** — they're what a cold
  start actually needs. Spend your specificity budget there. "Finish the loader"
  is useless; "`io/load.py:read_volume` still assumes isotropic spacing — handle
  anisotropic before wiring it into `train.py`" is a handoff.
- Pull the *substance* from the session's actual work and reasoning, not just the
  diff. The diff shows what changed; the DEVLOG should explain why and what's next.
- Keep it tight. A dense, skimmable entry beats an exhaustive one.

---

## Step 3 — Commit and push (Conventional Commits)

First look at what you're about to commit: `git status` and `git diff --stat`.
Make sure the DEVLOG update and any Step-1 version edits are included. If you see
files that shouldn't be tracked — `.env`, credentials, keys, large data/model
artifacts — and they aren't already gitignored, **stop and flag it** rather than
committing a secret or a 2 GB checkpoint.

### Message format

Use Conventional Commits: `type(scope): summary`, imperative mood, summary under
~72 chars, then an optional body with the details.

Common types: `feat` (new capability), `fix` (bug fix), `refactor`, `perf`,
`docs`, `test`, `build`, `ci`, `chore`.

Pick the type that reflects the **substantive** work of the session, not the
DEVLOG update. A session whose real work was a new feature is a `feat`, with the
devlog and version sync mentioned in the body — not a `docs` commit. If the
session's changes are cleanly separable (e.g. an unrelated bug fix plus a new
feature), prefer splitting into two logical commits over one mixed commit. If
they're entangled, one commit is fine.

**Example**
```
feat(recon): add anisotropic spacing support to volume loader

- read_volume now handles non-isotropic voxel spacing
- sync version to 0.4.1 across pyproject.toml and __init__.py
- log session notes in DEVLOG.md
```

### Pushing

Push to the current branch's upstream. If the branch has no upstream yet, set it:
`git push -u origin <branch>`.

Safety rails, always: **never force-push, never rewrite published history**
(no `--force`, no rebase of pushed commits, no amend-then-force), and never push
to a branch the user didn't intend. These are irreversible for collaborators; the
wrap-up routine is housekeeping, not history surgery. If a plain push is rejected
(e.g. the remote moved ahead), report it and ask — don't reach for `--force`.

---

## Finish

Close with a short, skimmable summary: the version that's now consistent, the
DEVLOG entry's title, the commit hash and message, and the branch you pushed to.
That's the receipt for a clean stopping point.
