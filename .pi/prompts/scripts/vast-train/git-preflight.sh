#!/usr/bin/env bash
# Validate the local code before resolving configs or renting an instance.
# Prints only the verified commit SHA on stdout; diagnostics go to stderr.
# --expected-commit keeps every combo in a sweep on the same revision.
set -euo pipefail

REMOTE_URL=https://github.com/skpro19/toy-act.git
BRANCH=act-v2
EXPECTED_COMMIT=

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --remote-url|--branch|--expected-commit)
      (($# >= 2)) && [[ -n "$2" ]] || fail "Missing value for $1"
      case "$1" in
        --remote-url) REMOTE_URL=$2 ;;
        --branch) BRANCH=$2 ;;
        --expected-commit) EXPECTED_COMMIT=$2 ;;
      esac
      shift 2
      ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

ROOT=$(git rev-parse --show-toplevel) || fail 'Run this check inside the project repository.'
cd "$ROOT"
LOCAL_BRANCH=$(git symbolic-ref --quiet --short HEAD) || fail "Detached HEAD; check out $BRANCH deliberately before training."
[[ "$LOCAL_BRANCH" == "$BRANCH" ]] || fail "Checked out $LOCAL_BRANCH; training requires $BRANCH. Switch branches deliberately."

STATUS=$(git status --porcelain=v1 --untracked-files=all)
if [[ -n "$STATUS" ]]; then
  printf '%s\n' "$STATUS" >&2
  fail 'Working tree is dirty. Commit, stash, or remove these changes deliberately before training. Ignored local configs and run state are allowed.'
fi

# Query the actual clone URL anonymously, not a possibly stale origin ref.
export GIT_TERMINAL_PROMPT=0
REMOTE_REFS=$(git -c credential.helper= ls-remote --exit-code "$REMOTE_URL" "refs/heads/$BRANCH") || fail "Cannot read $REMOTE_URL branch $BRANCH anonymously; check connectivity and public access. No stale-ref fallback is allowed."
read -r REMOTE_COMMIT REMOTE_REF <<< "$REMOTE_REFS"
[[ "$REMOTE_REF" == "refs/heads/$BRANCH" && "$REMOTE_COMMIT" =~ ^[0-9a-f]{40,64}$ ]] || fail 'Remote returned an invalid branch reference.'

# Fetch objects for ahead/behind diagnostics without moving the local branch.
# --no-write-fetch-head avoids sharing FETCH_HEAD across concurrent invocations.
git -c credential.helper= fetch --no-tags --no-write-fetch-head "$REMOTE_URL" "$REMOTE_COMMIT" >&2 || fail 'Cannot fetch the verified remote commit; stop before provisioning.'
LOCAL_COMMIT=$(git rev-parse HEAD)
if [[ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]]; then
  COUNTS=$(git rev-list --left-right --count "$LOCAL_COMMIT...$REMOTE_COMMIT")
  read -r AHEAD BEHIND <<< "$COUNTS"
  fail "Local HEAD $LOCAL_COMMIT differs from remote $BRANCH $REMOTE_COMMIT (ahead=$AHEAD, behind=$BEHIND). Sync deliberately before training; nothing was pulled, pushed, or reset."
fi
if [[ -n "$EXPECTED_COMMIT" && "$LOCAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  fail "Sweep pinned $EXPECTED_COMMIT but checkout and remote are now $LOCAL_COMMIT. Stop launching combos; existing runs and watchers must remain untouched."
fi
printf '%s\n' "$LOCAL_COMMIT"
