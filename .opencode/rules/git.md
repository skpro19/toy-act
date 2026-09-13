<!-- ---
description: "STRICT git commit/push policy with enforced commit message format"
alwaysApply: true
---

## Commit Messages
- ONLY create commits when the user explicitly asks.
- ONLY push when the user explicitly asks.
- Commit subject MUST be a single line and MUST follow Conventional Commits style:
  - Format: `<type>: <short summary>`
  - Allowed types include: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `build`, `perf`, `revert`
  - Use plain types only. Do NOT add scopes such as `feat(foo):`, `fix(bar):`, or `test(alohamini):`.
  - Summary after `type:` MUST start with a lowercase letter.
  - No trailing period.
  - Keep concise (prefer 6-12 words).
  - Prefer describing intent/outcome, not implementation details.
- Do NOT use vague subjects like `update`, `changes`, `fix stuff`, `misc`.
- Default to no commit body unless explicitly needed.
- Before finalizing a commit, check recent history and align style: `git log --oneline -10`.
- If a commit message does not match this format, rewrite it before pushing.

## Merge Commits
- Merge commits MUST also follow the format: `merge: <branch-name>`
- Example: `merge: inference-fix` (not "Merge branch 'inference-fix' into dev")
- Do NOT include the target branch name in the message ("into dev") — the branch structure is already implicit.

Examples:
- `feat: add profile-driven axis calibration config loading`
- `fix: prevent calibration overwrite unless toggle was pressed`
- `docs: clarify host-client branch sync workflow`


## Signing
- Always use signed commits: `git commit -S`
- At session start, verify identity matches GitHub: `git config user.name`, `user.email`, `user.signingkey`

## Pre-Work Sync
- Before significant code changes: `git fetch origin && git pull --ff-only`

<!-- ## Post-Push Sync
- After pushing from client, fast-forward all branches on host:
  ```
  ssh alohamini-pi "cd /home/skpro19/lerobot_alohamini && for b in \$(git branch | cut -c2-); do git checkout \$b && git pull --ff-only; done && git checkout <client_branch>"
  ```
- Verify sync: `git log -1 --format="%h %s" && ssh alohamini-pi "cd /home/skpro19/lerobot_alohamini && git log -1 --format='%h %s'"` --> -->
