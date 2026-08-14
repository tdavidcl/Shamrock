# Ensure Orphan Branch

GitHub composite action that creates a named orphan branch on the caller repository if it does not already exist.

## Usage

```yaml
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

## Inputs

| Name | Required | Description |
|------|----------|-------------|
| `branch_name` | No | Name of the orphan branch to ensure exists. Defaults to `orphan-branch`. |

## Requirements

- The caller must check out the repository first so `origin` is available.
- Creating the branch runs `git checkout --orphan` and `git rm -rf .` in the job workspace.
- Pushing the new branch uses the default `GITHUB_TOKEN`. The job needs `contents: write`.

## License

MIT — see [LICENSE](LICENSE).
