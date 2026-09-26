## Important Instructions
- **code readability** is paramount, do not compromise it for the sake of brevity or complexity

## General Instructions
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- refer `.opencode/rules/git.md` for git related instructions
- for `.canvas.tsx` files, follow `.cursor/rules/canvas.mdc` (design template based on `cs231n-multi-head-attention.canvas.tsx`)

## AWS Administration
- source `.env` or always use `AWS_PROFILE=toy-pickplace-backup` when running `aws` CLI commands for S3; the profile covers `s3://toy-pickplace` and prefix-scoped access to `s3://toy-act` (`checkpoints/act_v1`, `checkpoints/act_v2`, `runs/act_v1`, `runs/act_v2`, `datasets`), region is `ap-south-1`
- `s3://toy-act` access is prefix-scoped, so a prefix that is not in the policy (for example a new act version) is denied even though other prefixes work; `AccessDenied` on list or `403` on read almost always means the prefix is missing from the policy, not bad credentials
- `aws configure export-credentials --profile <profile> --format env-no-export` emits `KEY=VALUE` lines suitable for `uv run --env-file`; never print, commit, or transfer the output outside the workload
- when an application profile lacks permission to update its own IAM policy, use `aws login --profile <admin-profile> --region <region>` only after the user explicitly authorizes browser-based authentication
- verify the authenticated principal with `aws sts get-caller-identity --profile <admin-profile>` before making changes
- apply the narrowest required IAM policy from a reviewed JSON file; never place credentials, account-specific tokens, or browser-login URLs in repository files
- verify the resulting S3 operations using the workload profile, not the administrative profile
- run `aws logout --profile <admin-profile>` after verification so administrative browser credentials do not remain active

### Updating an existing inline IAM policy
1. list available profiles: `aws configure list-profiles`
2. log in with admin: `aws login --profile <admin-profile> --region <region>`
3. verify identity: `aws sts get-caller-identity --profile <admin-profile>`
4. list the user's inline policies: `aws iam list-user-policies --profile <admin-profile> --user-name <user>`
5. inspect the current policy document: `aws iam get-user-policy --profile <admin-profile> --user-name <user> --policy-name <policy>`
6. update the policy inline (keep existing statements, add new ones): `aws iam put-user-policy --profile <admin-profile> --user-name <user> --policy-name <policy> --policy-document file://reviewed-policy.json`
7. wait a few seconds for IAM propagation (inline policies can take ~10s)
8. verify with the workload profile: e.g. `aws s3 ls --profile <workload-profile>`
9. log out admin: `aws logout --profile <admin-profile>`

Notes:
- the workload user has a 2048-byte total limit across all inline policies; if a new inline policy would exceed it (`LimitExceeded` on `put-user-policy`), update an existing inline policy or attach a version to a managed policy (`aws iam create-policy-version <policy-arn> --policy-document file://reviewed.json --set-as-default`)
- verify write and delete too, not just list: a small throwaway `put-object`/`delete-object` under the new prefix confirms the full grant before starting a long run


## uv commands
- use `uv` instead of `pip`
- use `uv add` to add dependencies
- use `uv run` for Python scripts, tests, and CLIs instead of calling `python` directly

## Python Best Practices 
- do not use `dataclass` decorator
- do not use Ruff for linting or formatting
- use keyword-only arguments for functions/methods with multiple parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`)
- keep a function's return annotation on the `def` line when the whole signature fits one line (e.g. `def make_dagger_round_seeds(*, seed: int, rounds: int) -> list[int]:`)
- when a signature is broken across lines, put the closing `)`, `-> ReturnType`, and `:` on the **same line as the last parameter** — not on a separate line (e.g. `    epoch: int,) -> None:` with a trailing comma, or `    epoch: int) -> None:` without one); this keeps the full header collapsible as one block in the editor

## Vast.ai Administration
- API key is in `.env` as `VAST_API_KEY`
- The `vastai show user` CLI command is unreliable (returns 400); use the REST API instead
- Query balance via: `source .env && curl -sL -H "Authorization: Bearer $VAST_API_KEY" "https://console.vast.ai/api/v0/users/current"` — the `credit` field is the available balance

## Git Commits
- follow `.opencode/rules/git.md` for the full git policy; the rules below are the ones most often missed
- use conventional commits: `<type>: <short summary>` — one subject line only
- common types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `build`, `perf`, `revert`
- do **not** add a commit body unless the user explicitly asks for one
- summary after `type:` must start with a lowercase letter, have no trailing period, and stay concise (about 6–12 words)
- use plain types only — no scopes (`feat(scope): ...` is wrong)
- prefer intent/outcome over implementation detail; avoid vague subjects like `update`, `changes`, `misc`
- before committing, check recent style with `git log --oneline -10` and match it
- if the message is wrong, fix it **before** pushing; never push then amend/force-push unless the user explicitly asks to rewrite published history

Examples:
- `feat: add viewer camera capture and recalibrate top-camera`
- `fix: prevent calibration overwrite unless toggle was pressed`
- `docs: clarify host-client branch sync workflow`

## Sections to ignore
Do not treat these as code changes; ignore them during development and review:
- `docs/` — manually curated documentation; not a code change
- `ablations/` — experiment results and run tracking; not production code
- `ISSUES.md` — personal issue tracker; not a code change
- `IDEAS.md` — personal idea log; not a code change
