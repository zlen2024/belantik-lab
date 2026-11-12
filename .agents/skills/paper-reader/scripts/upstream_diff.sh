#!/usr/bin/env bash
# Show what changed upstream since the pinned submodule commits, limited to
# the files this skill was derived from. Read the diff, port what is useful
# into SKILL.md / references / scripts, then pin the new commits:
#
#   bash scripts/upstream_diff.sh            # review changes
#   git submodule update --remote            # move the pins to upstream HEAD
#   git add upstream && git commit           # and update UPSTREAM.md
set -euo pipefail
cd "$(dirname "$0")/.."

show() {
  local path=$1; shift
  local pinned
  pinned=$(git -C "$path" rev-parse HEAD)
  git -C "$path" fetch --quiet --depth 1 origin HEAD
  local latest
  latest=$(git -C "$path" rev-parse FETCH_HEAD)
  echo "=== $path"
  echo "pinned: $pinned"
  echo "latest: $latest"
  if [ "$pinned" = "$latest" ]; then
    echo "(no upstream changes)"
  else
    git -C "$path" diff --stat "$pinned" "$latest" -- "$@"
    git -C "$path" diff "$pinned" "$latest" -- "$@"
  fi
  echo
}

git submodule update --init --quiet
show upstream/paper-reading-skill SKILL.md README.md README_EN.md scripts
show upstream/nanochat .claude/skills/read-arxiv-paper
