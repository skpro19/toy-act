---
name: list-canvas
description: List all available Cursor canvases for the current project
disable-model-invocation: true
---

List all available `.canvas.tsx` files for the current project.

## Steps

1. Derive the managed canvases directory from the workspace path:
   - Take the absolute workspace root (e.g. `/media/skpro19/ssd/toy-act`)
   - Strip the leading `/`, replace remaining `/` with `-`
   - Managed dir: `~/.cursor/projects/<slug>/canvases/`

2. List native canvases in the managed directory:
   ```bash
   ls -1 ~/.cursor/projects/<slug>/canvases/*.canvas.tsx 2>/dev/null
   ```

3. List workspace-linked canvases (if the `canvas/` directory exists):
   ```bash
   ls -la canvas/*.canvas.tsx 2>/dev/null
   ```
   Resolve symlinks with `readlink -f` and note when a canvas lives in another project's managed directory.

4. For each canvas file, read the `<H1>` title near the default export. If no `<H1>` is found, use the filename without extension.

5. Output a grouped markdown list:

   **Managed canvases** — files in `~/.cursor/projects/<slug>/canvases/`:
   - `[filename](absolute-path)` — title

   **Workspace links** (only if `canvas/` has entries):
   - `[filename](workspace-path)` — title — symlink target or "local copy"

   If no canvases exist in either location, output: "No canvases found."

6. End with a one-line count: `Total: N canvas file(s)`.

Do not create, edit, or open any canvas — list only.
