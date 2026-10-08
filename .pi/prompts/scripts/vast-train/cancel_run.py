#!/usr/bin/env python3
"""Cancel one explicitly selected launched run; only its watcher destroys it.

No new launch or provisioning stage is run, and no terminal marker is fabricated.
The pinned runner receives TERM and writes its own terminal marker via its trap.
"""

import argparse
from pathlib import Path
import signal
import time

from iteration import Iteration, locate
from setup_run import ensure_watcher
from workflow import reconcile, watcher_alive
from workflow_common import Blocked, HELPERS, digest, instances, lock, redact, ssh_args

REMOTE_CANCEL = r'''
import hashlib
import os
from pathlib import Path
import signal
import subprocess

project = Path('/workspace/toy-act')
state = project / '.vast-train/state'
runner = project / '.pi/prompts/scripts/vast-train/runner.sh'
def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip() == expected_commit, 'Remote revision mismatch')
require((state / 'run-name').read_text().strip() == expected_run, 'Remote run identity mismatch')
if (state / 'completed').exists() or (state / 'failed').exists():
    print('Remote terminal marker already present; watcher owns cleanup')
else:
    require(hashlib.sha256(runner.read_bytes()).hexdigest() == expected_runner_hash, 'Runner script differs; refusing cancellation')
    pid = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', 'train', '#{pane_pid}'], text=True).strip())
    descriptor = os.pidfd_open(pid)
    try:
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        argv = [argument.decode() for argument in argv if argument]
        require(len(argv) == 2 and Path(argv[0]).name == 'bash' and argv[1] == str(runner), 'Train pane is not the expected runner')
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        print('TERM sent to verified training runner; waiting for its terminal marker and watcher cleanup')
    finally:
        os.close(descriptor)
'''


def cancel(*, iteration: Iteration, instance_id: int, wait_seconds: int) -> None:
    matches = [combo for combo in iteration.state['combos'] if combo.get('instance_id') == instance_id]
    if len(matches) != 1:
        raise Blocked('Instance must uniquely belong to the explicitly selected iteration')
    combo = matches[0]
    if not combo.get('training_launch_requested'):
        raise Blocked('Cancellation requires a launched run; this is not provisional cleanup')
    if reconcile(iteration=iteration, combo=combo):
        print(f'Instance {instance_id}: already terminal and reconciled')
        return
    run = combo['run']
    records = instances(iteration.journal)
    record = next((record for record in records if record['id'] == instance_id), None)
    if record is None or record.get('label') != run['label']:
        raise Blocked('Instance absent or label changed without a reconciled watcher report')
    ensure_watcher(journal=iteration.journal, combo=combo)
    if not watcher_alive(Path(combo['run_dir'])):
        raise Blocked('Saved watcher is not alive; refusing cancellation without cleanup ownership')
    script = (f'expected_commit = {iteration.manifest["git_commit"]!r}\n'
              f'expected_run = {combo["remote_run_name"]!r}\n'
              f'expected_runner_hash = {digest(HELPERS / "runner.sh")!r}\n' + REMOTE_CANCEL)
    iteration.journal.event(kind='cancellation_requested', combo=combo['id'], instance_id=instance_id)
    command = 'cd /workspace/toy-act && /root/.local/bin/uv run --frozen --only-group train python -'
    iteration.journal.run(args=ssh_args(run=run, command=command), input_text=script,
                          timeout=60, log_path=Path(combo['run_dir']) / 'cancellation.log')
    deadline = time.monotonic() + wait_seconds
    while True:
        if reconcile(iteration=iteration, combo=combo):
            print(f'Instance {instance_id}: watcher outcome={combo["status"]}; removal verified')
            return
        if time.monotonic() >= deadline:
            raise Blocked('Cancellation requested; watcher still owns cleanup. Retry this command to verify removal')
        iteration.journal.progress(text=f'Instance {instance_id}: waiting for watcher report and verified removal')
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True, help='explicit iteration ID or directory')
    parser.add_argument('--instance-id', type=int, required=True)
    parser.add_argument('--wait-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.instance_id <= 0 or not 0 <= args.wait_seconds <= 3600:
        parser.error('Invalid instance ID or wait budget')
    def interrupted(signum, frame, /) -> None:
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        directory = locate(identity=args.resume)
        with lock(directory / 'driver.lock'):
            cancel(iteration=Iteration(directory=directory), instance_id=args.instance_id, wait_seconds=args.wait_seconds)
        return 0
    except KeyboardInterrupt:
        print('Cancellation verifier interrupted; watcher and instances left untouched')
        return 130
    except (Blocked, ValueError, KeyError, TypeError, OSError) as error:
        print(f'Cancellation blocked: {redact(str(error))}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
