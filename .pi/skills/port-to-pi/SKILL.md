---
name: port-to-pi
description: Convert an opencode command or workflow (`.opencode/commands/*.md` plus its helper scripts) into a pi prompt template or skill. Use when porting opencode commands to pi.
---

# Port to pi

Port an opencode command to a pi resource without leaving any opencode dependency
behind. A port is done only when `grep -rin "opencode"` over the new resource
returns nothing.

## Choose the pi resource

| Use | Resource |
|---|---|
| Text-only prompt, no bundled files | Prompt template: `.pi/prompts/<name>.md` → `/<name>` |
| Needs helper scripts or long reference material | Skill: `.pi/skills/<name>/SKILL.md` + `scripts/` → `/skill:<name>` |
| Needs executable behavior (new tools, keybindings) | Extension: `.pi/extensions/<name>.ts` |

The filename (prompt) or `name` frontmatter (skill) becomes the command name.

## Mapping

| opencode | pi | Notes |
|---|---|---|
| `description:` frontmatter | same | Keep verbatim |
| `agent: <name>` frontmatter | none | Delete; pi runs with the session's active tools |
| `argument-hint` frontmatter | `argument-hint` | Optional; `<required>` / `[optional]` |
| skill frontmatter | `name` + `description` | `name` matches the directory; lowercase letters, digits, hyphens |
| `$ARGUMENTS`, `$@`, `$1..$N` | identical | pi also adds `${1:-default}`, `${@:N}`, `${@:N:L}` |
| `` !`cmd` `` | a bash step in the prompt | Not used in this repo |
| `@file` | read the file during execution | Not used in this repo |
| "interactive question tool" | the pi `question` tool | Requires the `.pi/extensions/question.ts` extension |
| "the interactive agent" | "the pi session" | Wording |

## File placement and scripts

| opencode path | pi path |
|---|---|
| `.opencode/commands/<name>.md` | `.pi/prompts/<name>.md` or `.pi/skills/<name>/SKILL.md` |
| `.opencode/commands/scripts/<group>/*` | `<resource>/scripts/<group>/*` |

- Copy the helper scripts into the pi resource. Do not edit or move the originals
  when the opencode workflow must stay intact; duplication is the accepted cost.
- Rewrite **every** `.opencode/...` reference, including paths embedded in remote
  commands that run after a clone (`/workspace/<repo>/.opencode/...` → the
  clone's new pi path). The scripts must be tracked so the clone contains them.

## Companion references

opencode commands often cross-reference each other
(`.opencode/commands/<other>.md`). Repoint those at the pi resource, or state
explicitly that the pi port supersedes them. Update both the referenced command
and every command that references it.

## Repo quirks

- Every current `.opencode/commands/*.md` uses `agent: build`.
- `.opencode/commands/scripts/vast-train/` is shared by the act v1/v2 train,
  resume, and rollout-sweep commands; a pi port of any of them needs a copy of
  the scripts it launches on the instance.
- `resume-tb-forwarding` is the only command using the interactive question tool.
- No command uses `` !`cmd` `` or `@file`, so no template-syntax translation is
  needed here.

## Procedure

1. Pick the resource type (prompt template vs skill) and target name.
2. Create the pi file and copy the opencode body into it.
3. Translate the frontmatter (drop `agent:`, add `argument-hint` or `name`).
4. Translate arguments, the question tool, and opencode wording.
5. Copy any helper scripts into the pi resource and rewrite all paths, including
   remote post-clone paths.
6. Repoint companion-command references.
7. Add a narrow `.gitignore` exception if `.pi/` is ignored.
8. Run the validation checklist.

## Validation

- `grep -rin "opencode"` over the new resource returns nothing.
- `grep -n "agent:"` over the new resource frontmatter returns nothing.
- Project is trusted; run `/reload`; the command/skill appears in the `/` menu
  (and `/skill:<name>` works for skills).
- `git check-ignore -v <path>` shows the intended tracked/ignored state.
- `git ls-files` shows every helper script so the instance clone contains it.
- No frontmatter warnings at startup.
