#!/usr/bin/env bash
# orient.sh — one-shot, read-only snapshot for the session-pickup routine.
#
# Prints the live git state and the newest DEVLOG.md entry so the model can
# orient fast at the start of a session. It ONLY reports: it never edits files,
# never commits, never pushes, and the only network it touches is a best-effort
# `git fetch` (so ahead/behind vs. upstream is accurate). If fetch fails
# (offline, no remote), it degrades gracefully and says so.
#
# Usage:
#   ./orient.sh [ROOT]     # ROOT defaults to the current directory
#
# Exit codes:
#   0 -> ran (a git repo or not; check the output)
#   2 -> ROOT is not a directory

set -uo pipefail

root="${1:-.}"
if ! cd "$root" 2>/dev/null; then
  echo "error: '$root' is not a directory" >&2
  exit 2
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

  echo "=== BRANCH ==="
  echo "${branch:-(detached HEAD)}"
  echo

  echo "=== WORKING TREE (git status --short) ==="
  status="$(git status --short 2>/dev/null)"
  if [ -n "$status" ]; then
    echo "$status"
    echo "(uncommitted changes present — someone worked here after the last log entry)"
  else
    echo "(clean)"
  fi
  echo

  echo "=== UPSTREAM DIVERGENCE ==="
  if git fetch --quiet 2>/dev/null; then
    :
  else
    echo "(git fetch skipped or failed — offline, or no remote; counts below may be stale)"
  fi
  if upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    # left = commits on upstream not local (behind); right = local not upstream (ahead)
    if counts="$(git rev-list --left-right --count "${upstream}...HEAD" 2>/dev/null)"; then
      behind="$(printf '%s' "$counts" | awk '{print $1}')"
      ahead="$(printf '%s' "$counts" | awk '{print $2}')"
      echo "upstream: ${upstream}   behind: ${behind}   ahead: ${ahead}"
      [ "${behind:-0}" -gt 0 ] && echo "(local checkout is behind — the remote moved ahead, e.g. a push from the cluster)"
      [ "${ahead:-0}" -gt 0 ] && echo "(local has unpushed commits)"
    fi
  else
    echo "(no upstream set for this branch)"
  fi
  echo

  echo "=== RECENT COMMITS (last 10) ==="
  git --no-pager log --oneline --decorate -n 10 2>/dev/null
  echo
else
  echo "Not a git repository at '$root'. Orient from the files directly instead."
  echo
fi

echo "=== DEVLOG (newest entry only) ==="
if [ -f DEVLOG.md ]; then
  # Print everything up to (but not including) the SECOND level-2 header, i.e.
  # the file title + the single newest "## YYYY-MM-DD — ..." entry.
  awk '
    /^## / { n++ }
    n == 2 { exit }
    { print }
  ' DEVLOG.md
else
  echo "(no DEVLOG.md at repo root — this may be a fresh project, or wrap-up"
  echo " has not run yet. Orient from git log and recently-changed files.)"
fi
