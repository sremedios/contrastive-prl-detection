---
name: runbook
description: >-
  Captures reusable bash commands and code snippets that come up during a coding
  session into a persistent, human-readable RUNBOOK.md at the repo root, so they
  can be re-run later — with minimal edits — without asking Claude to reconstruct
  them. The runbook is organised by purpose (not chronology), holds the current
  canonical form of each command, and marks the values the user must fill in.
  Use this whenever the user wants to keep a command or snippet for later —
  phrasings like "save that command", "add this to the runbook", "log this
  snippet", "note that so I can rerun it", "keep this for next time", "I don't
  want to ask you for this again", or "track the commands we've been using". Also
  offer it proactively — briefly, once — when you hand over a non-trivial command
  the user will clearly need again (a cluster run, a data-prep pipeline, an
  rsync/scp transfer, an environment rebuild), especially a command they run in
  an environment this session can't reach and therefore can't reproduce on its
  own. Trigger even when the user names only one command ("save just this one"),
  because the point is to catch reusable commands before they scroll away.
---

# Runbook

A persistent home for the commands and snippets that come up while working on a
project but don't naturally live in a tracked file — the cluster job you submit
by hand, the preprocessing one-liner, the rsync that pulls results back, the
environment rebuild you run once a month and always half-remember.

It exists because these commands are the easiest useful thing to lose. They
scroll out of the conversation, and the next time you need one you either dig
through old logs or ask Claude to reconstruct it from scratch — and a
reconstructed command is a *guess* at what you actually ran, not the thing
itself. The runbook keeps the real command, copy-paste-ready, so re-running it is
a lookup, not a re-derivation.

This pairs with `session-wrapup` and `session-pickup`. Those maintain the DEVLOG
(the narrative of *what happened*); the runbook maintains the commands (the
*how-to-do-it-again*). A good split to hold in mind: the DEVLOG is prose you read
top-to-bottom to catch up; the runbook is a reference you jump into to grab one
block and run it.

If you're not in a git repository, you can still write `RUNBOOK.md` — it's just a
file — but say so, since it won't be version-tracked and the "history over time"
below leans on git.

---

## The artifact: `RUNBOOK.md` at the repo root

One markdown file at the repo root, organised **by purpose**, not by date. You
look a command up by *what it does* ("the one that rebuilds the dataset"), never
by *when it was added*, so purpose-grouping is what makes it fast to use.

Group entries under headings that match the project's natural task areas —
`Environment`, `Data`, `Training / cluster`, `Analysis`, `Local dev`, or whatever
fits. If the project is small enough that a flat list is clearer than sections, a
flat list is fine; add sections only once they earn their keep.

### What belongs

- **Reusable commands** — anything you'll plausibly run more than once, or run
  again after a gap long enough that you'd have forgotten the exact invocation.
- **Hand-off commands especially** — the ones Claude writes for the user to run
  in an environment it can't reach (cluster, GPUs, the work computer). These are
  the highest value: Claude never sees them run, so if they aren't captured
  they're gone, and they usually carry fiddly flags and paths that are painful to
  reconstruct.
- **Small code snippets** that stand alone and get reused — a one-off analysis
  cell, a data-massaging Python snippet, a query — where promoting them to a
  committed script would be overkill but losing them would be annoying.

### What doesn't

- **Trivial commands.** `cd`, `ls`, `git status` — nobody needs a saved copy.
- **Throwaway debugging.** The command you ran once to see what was wrong and will
  never run again is noise here.
- **One-time manual interventions.** A hand-applied data fix or a
  never-again migration is *history*, not a reusable command — it belongs in the
  DEVLOG (that's what `session-wrapup` records), not the runbook. Keeping the
  runbook to genuinely re-runnable things is what keeps it trustworthy: everything
  in it is something you can actually run again.
- **Secrets.** Never write credentials, tokens, API keys, connection strings with
  embedded passwords, or private hostnames into the runbook — it's a committed
  file. If a command needs one, mark it as a placeholder (see below) and keep the
  real value out.

When in doubt, ask the one-line question: *would I want to run this again without
re-typing it?* If yes, it belongs.

---

## Entry format

Each entry is a short titled block: what it does, the command verbatim, and the
values the user must supply. Keep the surrounding prose to a line — the command
is the content.

````markdown
# Runbook — <project name>

Copy-paste-ready commands for this project. Organised by purpose. Each holds the
current canonical form; fill the marked placeholders before running.

## <Section — e.g. Training / cluster>

### <Imperative title — what running this does>
<One line: when or why you run this, and what a successful run looks like if that
isn't obvious.>

```bash
<the command, verbatim, with values-to-fill marked as <YOUR_PLACEHOLDER>>
```

_Fill_: `<YOUR_DATA_ROOT>` — path to the dataset on the cluster ·
`<PARTITION>` — Slurm partition to submit to
_Updated YYYY-MM-DD_: <only when a change would surprise future-you, and why>
````

Guidance:

- **Verbatim command, real flags.** The value of the runbook is that the command
  is the one that actually worked. Don't paraphrase it into something cleaner
  that you haven't run.
- **Mark what the user fills**, using the `<YOUR_DATA_ROOT>` convention shared with
  `session-pickup` so nothing runs half-configured. List the placeholders under
  `_Fill_` with a word on what each is. If there are none, omit the line.
- **Title by effect, not mechanism.** "Rebuild the preprocessed volume cache" is
  findable; "Run prep.py" is not.
- **Say what success looks like** for anything long or risky — the last line to
  expect, the file that should appear — so a re-run can be told apart from a
  silent failure. Skip it when the command is obviously all-or-nothing.
- **Omit empty lines.** No `_Fill_: none`, no empty `_Updated_`. Empty scaffolding
  is noise; drop it.

---

## Keeping it over time

The runbook holds the **current canonical** form of each command — the one thing
you should run today. That's what makes it a reference rather than an archive.

- **Update in place; don't append duplicates.** When a command changes — a new
  flag, a renamed config, a corrected path — edit the existing entry rather than
  adding a second near-identical block. Two copies of "the training command" is
  exactly the confusion the runbook is meant to remove. Before adding an entry,
  scan the relevant section for one that already covers this command and update
  that instead.
- **Git is the history.** Because `RUNBOOK.md` is committed, `git log -p
  RUNBOOK.md` is the full record of how every command evolved. You don't need to
  keep old versions inline to "track over time" — the file stays lean and the
  history stays complete, each in its right place.
- **Note a change only when it's surprising.** Add a one-line `_Updated
  YYYY-MM-DD_: ...` only when a future reader would otherwise wonder *why* the
  command looks the way it does — the same test the ADR format uses. A routine
  path tweak needs no note; "switched to `--devices 4` because 8 OOMs on the new
  volumes" earns one.

---

## Routine

1. **Locate or create `RUNBOOK.md`** at the repo root. If it doesn't exist, create
   it with the title header and the one-line intro from the template.
2. **Identify the command(s) to capture** — either the ones the user named, or,
   when triggered at wrap-up or offered proactively, the reusable commands that
   came up this session (skip the trivia and throwaways per *What doesn't*).
3. **Check for an existing entry** in the matching section. If one covers this
   command, update it in place; otherwise add a new entry in the right section
   (creating the section if needed).
4. **Scrub for secrets.** Replace any credential, token, key, or private host with
   a marked placeholder before writing. If a command can't be made safe as a
   placeholder, say so and leave it out rather than committing a secret.
5. **Write the entry** in the format above — verbatim command, marked placeholders,
   success cue if warranted.
6. **Report** what you captured: the section, the entry title(s), and a one-line
   note on anything you deliberately left out (a secret you placeholdered, a
   throwaway you skipped) so the user can correct the call. If you're in a git
   repo, the entry ships with the next commit — `session-wrapup` will carry it —
   so there's nothing separate to push here unless the user asks.

---

## Composes with the session skills

- **At wrap-up.** When `session-wrapup` runs, flushing this session's reusable
  commands into the runbook is a natural part of leaving a clean stopping point —
  the DEVLOG gets the narrative, the runbook gets the commands, and both ride the
  same commit.
- **At pickup.** `session-pickup` can skim `RUNBOOK.md` alongside `CONTEXT.md` so a
  cold session already knows the established commands — and can hand one straight
  back to the user (under its "you write, they run" boundary) instead of inventing
  a fresh invocation for a task the project already has a known command for.
