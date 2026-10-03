#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
umask 077

LOCK_FILE=/tmp/toy-act-local-wrapper.lock

usage() {
  echo "Usage: $0 allocate RUN_NAME | restore RUN_NAME INDEX | release-unstarted RUN_NAME INDEX" >&2
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
  local index ssh_session tb_session tb_port owner_file

  validate_run_name "$run_name"
  for index in $(seq 0 999); do
    ssh_session="act-ssh-$index"
    tb_session="act-tb-$index"
    tb_port=$((6006 + index))
    owner_file="/tmp/toy-act-local-wrapper-$index.owner"
    if tmux has-session -t "$ssh_session" 2>/dev/null ||
      tmux has-session -t "$tb_session" 2>/dev/null ||
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

restore() {
  local run_name="$1" index="$2"
  local owner_file ssh_session tb_session tb_port
  validate_run_name "$run_name"
  case "$index" in
    ""|*[!0-9]*) echo 'ERROR: invalid workflow index' >&2; return 2 ;;
  esac
  test "$index" -le 999 || { echo 'ERROR: index exceeds lease range' >&2; return 2; }
  owner_file="/tmp/toy-act-local-wrapper-$index.owner"
  ssh_session="act-ssh-$index"
  tb_session="act-tb-$index"
  tb_port=$((6006 + index))
  if [ -e "$owner_file" ]; then
    test "$(<"$owner_file")" = "$run_name" || {
      echo 'ERROR: recorded forwarding lease belongs to another run' >&2
      return 1
    }
  else
    if tmux has-session -t "$ssh_session" 2>/dev/null ||
      tmux has-session -t "$tb_session" 2>/dev/null ||
      ss -ltn "sport = :$tb_port" | grep -q LISTEN; then
      echo 'ERROR: recorded forwarding slot is occupied without matching ownership' >&2
      return 1
    fi
    printf '%s\n' "$run_name" > "${owner_file}.tmp.$$"
    mv "${owner_file}.tmp.$$" "$owner_file"
  fi
  printf '%s\n' "$index" "$ssh_session" "$tb_session" "$tb_port"
}

release_unstarted() {
  local run_name="$1"
  local index="$2"
  local owner_file

  validate_run_name "$run_name"
  case "$index" in
    ""|*[!0-9]*)
      echo "ERROR: INDEX must be a non-negative integer" >&2
      return 2
      ;;
  esac

  owner_file="/tmp/toy-act-local-wrapper-$index.owner"
  test -f "$owner_file" || {
    echo "ERROR: local wrapper lease $index does not exist" >&2
    return 1
  }
  test "$(<"$owner_file")" = "$run_name" || {
    echo "ERROR: local wrapper lease $index belongs to another run" >&2
    return 1
  }

  tmux kill-session -t "act-ssh-$index" 2>/dev/null || true
  tmux kill-session -t "act-tb-$index" 2>/dev/null || true
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
  restore)
    test "$#" -eq 2 || usage
    restore "$1" "$2"
    ;;
  release-unstarted)
    test "$#" -eq 2 || usage
    release_unstarted "$1" "$2"
    ;;
  *) usage ;;
esac
