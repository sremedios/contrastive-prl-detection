---
name: session-pickup
description: >-
  Start-of-session orientation routine for a coding project — the bookend to
  session-wrapup. Reads the newest DEVLOG.md entry, reconciles it against the
  live git state (branch, uncommitted work, commits and upstream divergence since
  that entry), skims CONTEXT.md and the most recent ADRs if they exist, then
  briefs the user on where things stand and proposes the next move before diving
  in. It also (re)establishes the standing execution boundary for the whole
  session: the real datasets, GPUs, and heavy runs live on a remote cluster /
  work computer this session cannot reach, so hand the user copy-paste-ready
  commands for anything that needs them and never fabricate their output —
  while running self-contained *local* checks (unit tests, type checks, linters,
  small scripts over repo-local files) directly. Use whenever the user is picking
  a project back up — phrasings like "let's keep going from last time", "pick up
  where we left off", "check the logs and orient yourself", "catch me up",
  "where were we", "resume", "let's continue", or the first substantive message of
  a fresh session on an existing repo. Trigger it even when the user asks for only
  part of this (just "check the log" or just "where did we leave off"), because at
  session start the full orientation is almost always what helps.
---

# Session pickup

A fixed routine for cleanly *resuming* work on a code project — the mirror image
of `session-wrapup`. Wrap-up leaves a clean stopping point (a DEVLOG entry, a
consistent version, a pushed commit); pickup uses that stopping point to get a
cold session productive fast, without the user having to re-explain everything
they said yesterday.

It exists to replace the paragraph the user would otherwise type by hand every
morning — "read the logs, orient yourself, and by the way remember you can't run
things on my cluster." Run it and that context is loaded automatically.

Run the steps in order: **(1) load the standing execution boundary, (2) snapshot
the state, (3) read the DEVLOG as a handoff, (4) reconcile it against git,
(5) reload the domain language, (6) brief and propose the next move.** The first
step governs the whole session, not just pickup — read it before touching
anything else.

If you're not in a git repository, say so; you can still orient from `DEVLOG.md`
and the files, but skip the git-specific steps.

---

## Step 1 — Load the execution boundary (this governs the whole session)

**The user runs the real work; you don't have their environment.** The primary
data and the code that operates on it — training runs, large preprocessing,
anything touching real datasets, GPUs, licensed software, the cluster scheduler,
or internal network resources — happen on a remote cluster or the user's work
computer that **this session cannot access.** Internalize this now so it shapes
every suggestion you make today, not just the first one.

The boundary is a division of labor, not a refusal to be useful:

- **Needs the user's environment → you write, they run.** Produce the exact
  command(s) as a single copy-paste-ready block, then **stop and wait** for the
  user to run them and paste back the output. Do not imagine what the output
  would be, do not proceed on an assumed result, and do not narrate a run as if
  it happened. The pasted-back logs are ground truth; reconcile your plan against
  what actually comes back.
- **Self-contained and local → just do it.** Unit tests, type checks, linters,
  a quick refactor you can verify against files already in the repo, a small
  script over repo-local data — run these yourself. The user explicitly welcomes
  local unit-test work; don't hand back a command for something you can just run.
- **Unsure which side it's on → assume it needs their environment and ask.**
  Handing over a command the user didn't need costs them one paste; fabricating a
  result they can't see costs them a debugging session built on fiction.

### Writing commands the user will actually run

Make hand-off commands paste-ready: assume the repo root as the working directory
(or state the directory explicitly), avoid dangling placeholders, and if a value
is genuinely the user's to supply (a partition name, a checkpoint path, a node
count) mark it clearly, e.g. `--data-root <YOUR_DATA_ROOT>`, so nothing runs half-
configured. When a step is long or risky, say what a correct run looks like so the
user can tell success from a silent failure before pasting results back.

**Example — a run the user must execute:**
> This needs the cluster's GPUs and the full dataset, so run this and paste back
> the last ~20 lines:
> ```bash
> python -m train.fit --config configs/aniso.yaml --data-root <YOUR_DATA_ROOT> \
>   --devices 4 2>&1 | tee logs/aniso_$(date +%Y%m%d).log
> ```

**Example — a check you run yourself:**
> The spacing math is pure and testable without any data, so I'll just run the
> unit tests locally. *(runs `pytest tests/test_spacing.py -q` directly)*

---

## Step 2 — Snapshot the state

Get an objective picture before forming any narrative. Run the bundled snapshot
from the repo root:

```bash
bash <skill_dir>/scripts/orient.sh .
```

It's **read-only** — it reports and never edits, commits, or pushes (its only
network touch is a best-effort `git fetch` so ahead/behind counts are accurate).
It prints the current branch, the working-tree status, upstream divergence, the
last ten commits, and the **newest DEVLOG.md entry** — the four things you need to
reconcile "what the log says" against "what the repo actually looks like."

If you'd rather run the pieces by hand, the equivalents are `git status -sb`,
`git log --oneline -n 10`, `git fetch && git rev-list --left-right --count @{u}...HEAD`,
and reading the top entry of `DEVLOG.md`.

---

## Step 3 — Read the DEVLOG entry as a handoff

The newest entry sits at the top of `DEVLOG.md` (that's wrap-up's invariant).
Read it as what it is: **a handoff written for a competent colleague with no
memory of last session.** You are that colleague today.

Weight the sections by how much they unblock you:

- **Next steps** and **Notes & gotchas** are the highest-value parts — they're
  the reason the log exists. The next action and the traps to avoid usually live
  here. Read them most carefully.
- **State** tells you what's verified vs. in-progress vs. broken — so you don't
  waste time re-checking something already known-good or build on something
  known-broken.
- **Summary**, **Changes**, and **Decisions & rationale** give you the "why" so
  you don't relitigate a question that was already settled last session.

Pull out, concretely: the intended **next action**, the **version** and
**branch** the last session was on, and any **gotcha** that would bite a cold
start. You'll check those against reality in the next step.

If there's **no DEVLOG.md**, don't stall. Orient from `git log`, the branch, and
the most recently changed files, tell the user the log is missing, and note that
`session-wrapup` will create it at the end of this session.

---

## Step 4 — Reconcile the log against reality

The DEVLOG is a narrative someone wrote at a stopping point; **git is the ground
truth for what the repo is right now.** Between then and now the user may have
committed from the cluster, left work uncommitted on the work computer, or
switched branches. When the two disagree, trust git and surface the gap rather
than silently trusting the log.

Check for the discrepancies that actually change what you should do next:

- **Branch mismatch.** If the entry says `Branch: feat/loader` but you're on
  `main`, you may be looking at a different line of work than where the log left
  off. Point it out before building on the wrong base.
- **Dirty working tree.** Uncommitted changes mean someone worked here *after*
  the last entry — likely the user on their machine. Don't build on top of a
  mystery diff; look at what changed and confirm what it is first.
- **Upstream divergence.** *Behind* upstream means the local checkout is stale
  (the remote moved ahead — e.g. a push from the cluster); offer to pull, but
  **never** force or rebase pushed history — that's wrap-up's rule too, and it's
  irreversible for collaborators. *Ahead* means there are unpushed local commits;
  note them so they aren't forgotten.
- **Version drift.** If the version the log recorded no longer matches the code,
  a change landed out of band. Flag it; don't quietly assume the log is right.

Resolve anything that changes the picture (a dirty tree, a wrong branch) **before**
proposing to charge ahead — starting work on a stale or unexpected base is the
one mistake a cold start most easily makes.

---

## Step 5 — Reload the domain language (if it exists)

If the repo carries the artifacts the `grill-with-docs` skill maintains, reload
them so you speak the project's language from the first sentence instead of
drifting into synonyms:

- **`CONTEXT.md`** (or the contexts named in a root **`CONTEXT-MAP.md`**) — skim
  the glossary so you use the project's canonical terms and honor its `_Avoid_`
  list. Getting the vocabulary right immediately is what makes a resumed session
  feel continuous rather than restarted.
- **`docs/adr/`** — glance at the most recent one or two ADRs for decisions that
  constrain today's work, so you don't propose something already deliberately
  ruled out.

Keep this light: you're refreshing your memory of the shared language, not
auditing the docs. If none of these files exist, skip the step.

---

## Step 6 — Brief and propose the next move

Close pickup the way wrap-up opens with a receipt — a tight, skimmable
orientation the user can confirm or redirect in one read:

- **Where we are:** branch and version, plus a one-line "last session left off
  doing X."
- **Anything off:** dirty tree, branch mismatch, behind/ahead upstream, version
  drift — only if present. Silence here should mean "clean."
- **Proposed next move:** the concrete next action, taken from the DEVLOG's Next
  steps and reconciled with what git actually shows — named down to the file or
  function where you can (a resumable handoff, per wrap-up, reads like
  "`io/load.py:read_volume` still assumes isotropic spacing," not "finish the
  loader"). If that next move needs the cluster, say so and have the command
  ready per Step 1.

Then **confirm before diving in** — one short check, not an interrogation. The
user may want to point the day somewhere the log didn't anticipate, and it's
cheaper to hear that now than after you've started. Once they confirm, get to
work under the Step 1 boundary: run local checks yourself, hand over anything
that needs their environment, and never invent a result you can't see.
