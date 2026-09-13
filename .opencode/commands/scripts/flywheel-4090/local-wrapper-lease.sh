#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

LOCK_FILE=/tmp/toy-pickplace-flywheel-local-wrapper.lock

usage() {
  echo "Usage: $0 allocate RUN_NAME | release-unstarted RUN_NAME INDEX" >&2
  exit 2
}

validate_run_name() {
  case "$1" in
    ""|*[!A-Za-z0-9._-]*)
      echo "ERROR: invalid RUN_NAME" >&2
      exit 2
      ;;
  esac
}

allocate() {
  local run_name="$1"
  local index ssh_session tb_session supervisor_session tb_port owner_file

  validate_run_name "$run_name"
  for index in $(seq 0 999); do
    ssh_session="vast-ssh-$index"
    tb_session="tb-flywheel-$index"
    supervisor_session="flywheel-supervisor-$index"
    tb_port=$((6006 + index))
    owner_file="/tmp/toy-pickplace-flywheel-$index.owner"
    if tmux has-session -t "$ssh_session" 2>/dev/null ||
      tmux has-session -t "$tb_session" 2>/dev/null ||
      tmux has-session -t "$supervisor_session" 2>/dev/null ||
      test -e "$owner_file" ||
      ss -ltn "sport = :$tb_port" | grep -q LISTEN; then
      continue
    fi

    printf '%s\n' "$run_name" > "${owner_file}.tmp.$$"
    mv "${owner_file}.tmp.$$" "$owner_file"
    printf '%s\n' "$index" "$ssh_session" "$tb_session" "$tb_port"
    return 0
  done

  echo "ERROR: no free local workflow index" >&2
  return 1
}

release_unstarted() {
  local run_name="$1"
  local index="$2"
  local owner_file supervisor_session

  validate_run_name "$run_name"
  case "$index" in
    ""|*[!0-9]*)
      echo "ERROR: INDEX must be a non-negative integer" >&2
      return 2
      ;;
  esac

  owner_file="/tmp/toy-pickplace-flywheel-$index.owner"
  supervisor_session="flywheel-supervisor-$index"
  test -f "$owner_file" || {
    echo "ERROR: local wrapper lease $index does not exist" >&2
    return 1
  }
  test "$(<"$owner_file")" = "$run_name" || {
    echo "ERROR: local wrapper lease $index belongs to another run" >&2
    return 1
  }
  if tmux has-session -t "$supervisor_session" 2>/dev/null; then
    echo "ERROR: refusing to release lease $index while its supervisor is active" >&2
    return 1
  fi

  tmux kill-session -t "vast-ssh-$index" 2>/dev/null || true
  tmux kill-session -t "tb-flywheel-$index" 2>/dev/null || true
  rm -f "$owner_file"
}

test "$#" -ge 1 || usage
action="$1"
shift

exec 9>"$LOCK_FILE"
flock 9
case "$action" in
  allocate)
    test "$#" -eq 1 || usage
    allocate "$1"
    ;;
  release-unstarted)
    test "$#" -eq 2 || usage
    release_unstarted "$1" "$2"
    ;;
  *) usage ;;
esac
