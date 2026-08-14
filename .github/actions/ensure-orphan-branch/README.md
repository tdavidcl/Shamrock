# action-ensure-orphan-branch

Create a named orphan branch if it is missing. If the branch already exists, the action does nothing.

The calling job must check out the repository first so `origin` is available. Creating the branch runs `git checkout --orphan` and `git rm -rf .` in that workspace.

## Permissions

```yaml
permissions:
  contents: write
```

## Usage

### Same repository (before the split)

```yaml
name: Ensure Orphan Branch

on:
  workflow_dispatch:
  push:
    branches:
      - main

jobs:
  ensure-orphan-branch:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/ensure-orphan-branch
        with:
          branch_name: orphan-branch
```

### Standalone repository (after the split)

```yaml
name: Ensure Orphan Branch

on:
  workflow_dispatch:
  push:
    branches:
      - main

jobs:
  ensure-orphan-branch:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: Shamrock-code/action-ensure-orphan-branch@v1
        with:
          branch_name: orphan-branch
```

Pin a commit SHA in production, matching other Shamrock extracted actions.

## Inputs

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `branch_name` | no | `orphan-branch` | Branch to create when missing. |

## Outputs

| Output | Description |
| --- | --- |
| `exists` | `true` if the branch already existed. |

## Splitting into a standalone repository

Create `Shamrock-code/action-ensure-orphan-branch` and copy `action.yml` and `README.md`. Then:

1. Add an MIT `LICENSE`, same as `action-trigger-ci-empty-commit`.
2. Tag `v1`.
3. Replace callers with `uses: Shamrock-code/action-ensure-orphan-branch@<sha>`.
