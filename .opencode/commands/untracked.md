---
description: List all untracked and unstaged files for the current branch
agent: build
---

List all untracked, unstaged, and staged files for the current git branch using `git status --porcelain`. Show files grouped by their status prefixes:
- `??` = untracked (new) files
- ` A` = added to index but unstaged
- `A ` = added to index (staged)
- ` M` = modified but unstaged
- `M ` = modified in index (staged)
- ` D` = deleted but unstaged
- `D ` = deleted in index (staged)
- `R ` = renamed in index
- `C ` = copied in index

If no changes exist, output: "No changes found."
