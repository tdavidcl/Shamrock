# PR Steward — repo-specific overrides

This file is repo-specific guidance for whichever Claude session is
babysitting/stewarding a Shamrock PR. It takes precedence over the generic
"CI red" drive-to-green loop for the one case below.

## Mergify merge-queue CI: cancel on phystest-pipeline failure

How to tell a run is a Mergify merge-queue run: the `on PR` workflow
(`.github/workflows/on_pr.yml`) sets `is_merge_queue: true` into the
reusable `CI` workflow (`main_workflow.yml`) whenever the PR's head branch
matches `mergify/merge-queue/*`. Concretely: the workflow run's head branch
starts with `mergify/merge-queue/`.

If, on such a run, any of these jobs fails:

- `build_push_docker` ("Build dockerfiles")
- `make_documentation` ("Doc")
- `shamrock_linux_acpp_phystests` ("Tests (Phys)")

cancel the whole `on PR` workflow run right away
(`mcp__github__actions_run_trigger`, method `cancel_workflow_run`) instead
of diagnosing and pushing a fix.

Why: `.mergify.yml` sets `max_parallel_checks: 1`, so one running
merge-queue CI blocks every other queued PR until it finishes, times out,
or is cancelled. The run's head is a synthetic merge commit Mergify
manages, not a branch you can push a fix to — there is nothing to fix
forward here, only time being wasted for the whole queue. Cancelling
promptly lets Mergify dequeue this PR and move on to the next batch.

This override applies only to merge-queue runs. On a regular `pull_request`
or `push` run, a failure in these same jobs still goes through the normal
CI-red drive-to-green loop (root-cause it, push a fix, or explain why not).
