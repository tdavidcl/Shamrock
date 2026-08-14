# action-ensure-orphan-branch

Create a named orphan branch if it is missing. If the branch already exists, the action is a no-op and does not rewrite history.

This is a composite action. While it lives in Shamrock it is invoked by relative path. After it is split into `Shamrock-code/action-ensure-orphan-branch`, pin a tag or commit SHA instead.

The action does **not** check out or delete the job workspace. It creates the orphan commit in a temporary repository, so it is safe to run alongside other steps.

## Permissions

The calling job must grant write access to git contents:

```yaml
permissions:
  contents: write
```

`GITHUB_TOKEN` is enough for the current repository. Pass a PAT (for example `${{ secrets.GHACTION_PAT }}`) when:

- the target repository is not the current one
- org or repository policy blocks `GITHUB_TOKEN` from creating branches
- branch protection requires a different identity

## Usage

### Same repository (before the split)

Checkout is required only so GitHub can load the local `action.yml`:

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
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Ensure orphan branch
        uses: ./.github/actions/ensure-orphan-branch
        with:
          branch_name: orphan-branch
```

### Standalone repository (after the split)

Do not check out the caller repository. The runner fetches the action from its own repo:

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
      - name: Ensure orphan branch
        uses: Shamrock-code/action-ensure-orphan-branch@v1
        with:
          branch_name: orphan-branch
```

Pin a commit SHA in production, matching other Shamrock extracted actions.

### Custom token, repository, or commit message

```yaml
- uses: Shamrock-code/action-ensure-orphan-branch@v1
  with:
    branch_name: orphan-branch
    github_token: ${{ secrets.GHACTION_PAT }}
    repository: owner/name
    commit_message: Initial commit on orphan branch
```

## Inputs

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `branch_name` | no | `orphan-branch` | Branch to create when missing. |
| `github_token` | no | `${{ github.token }}` | Token used for `ls-remote` and `push`. |
| `repository` | no | `${{ github.repository }}` | Target repository (`owner/name`). |
| `commit_message` | no | `Initial commit on orphan branch` | Message for the initial orphan commit. |
| `git_user_name` | no | `github-actions[bot]` | Git author name. |
| `git_user_email` | no | `41898282+github-actions[bot]@users.noreply.github.com` | Git author email. |

## Outputs

| Output | Description |
| --- | --- |
| `branch_name` | Branch that was checked or created. |
| `exists` | `true` if the branch already existed before this run. |
| `created` | `true` if this run created the branch. |

```yaml
- uses: ./.github/actions/ensure-orphan-branch
  id: orphan
  with:
    branch_name: orphan-branch

- run: |
    echo "branch=${{ steps.orphan.outputs.branch_name }}"
    echo "existed=${{ steps.orphan.outputs.exists }}"
    echo "created=${{ steps.orphan.outputs.created }}"
```

## Behavior

1. Validates `branch_name` with `git check-ref-format`.
2. Queries `refs/heads/<branch_name>` with `git ls-remote` (exact ref, not a substring grep).
3. If the ref exists, sets `exists=true`, `created=false`, and exits.
4. If it is missing, `git init -b` in a temp directory, commits `README.md` containing `# <branch_name>`, and pushes `HEAD:refs/heads/<branch_name>`.
5. If two jobs race and the push fails because the branch appeared, the action treats that as success (`exists=true`, `created=false`).

The resulting branch has a single root commit (no parent) and does not contain files from `main`.

## Splitting into a standalone repository

Create `Shamrock-code/action-ensure-orphan-branch` and copy:

- `action.yml`
- `ensure-orphan-branch.sh`
- `README.md`
- `test_ensure_orphan_branch.sh` (optional)

Then:

1. Add an MIT `LICENSE`, same as `action-trigger-ci-empty-commit`.
2. Tag `v1`.
3. Replace callers with `uses: Shamrock-code/action-ensure-orphan-branch@<sha>`.
4. Drop the `actions/checkout` step that existed only to resolve the local action.

## Local test

```bash
.github/actions/ensure-orphan-branch/test_ensure_orphan_branch.sh
```

The script talks to a temporary local bare remote; it does not need GitHub credentials.
