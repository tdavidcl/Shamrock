#!/usr/bin/env bash
# Hang until every job that On Push / On Schedule run on main has completed
# in this PR run. Uses `gh run view --json jobs` and treats queued /
# in_progress (and not-yet-created) jobs as not done.
#
# The parent run status (`gh run view "$GITHUB_RUN_ID" --json status`) stays
# in_progress while this job is running, so it cannot be used as the signal.

set -euo pipefail

: "${GITHUB_RUN_ID:?}"
: "${GITHUB_REPOSITORY:?}"

GH_REPO="${GH_REPO:-$GITHUB_REPOSITORY}"
export GH_PAGER=cat
POLL_SECONDS="${POLL_SECONDS:-15}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
MIN_CI_NESTED_JOBS="${MIN_CI_NESTED_JOBS:-20}"
MAX_REFERENCE_AGE_SECONDS="${MAX_REFERENCE_AGE_SECONDS:-1209600}"

should_ignore() {
    case "$1" in
        all | all_light | wait_for_push_schedule | CI)
            return 0
            ;;
        "Notify Light CI" | "Check Codecov secret" | "Check if scheduled CI is needed" | \
            "Wait for On Push and On Schedule jobs")
            return 0
            ;;
    esac
    return 1
}

pick_reference_run() {
    local workflow="$1"
    local now id created age count
    now="$(date -u +%s)"

    while IFS=$'\t' read -r id created; do
        age="$((now - $(date -d "$created" +%s)))"
        if [ "$age" -gt "$MAX_REFERENCE_AGE_SECONDS" ]; then
            continue
        fi
        count="$(
            gh run view "$id" --repo "$GH_REPO" --json jobs \
                --jq '[.jobs[].name | select(startswith("CI / "))] | length'
        )"
        echo "reference ${workflow} run ${id} age=${age}s ci_jobs=${count}" >&2
        if [ "$count" -ge "$MIN_CI_NESTED_JOBS" ]; then
            echo "$id"
            return 0
        fi
    done < <(
        gh run list --repo "$GH_REPO" --workflow "$workflow" --branch main --limit 30 \
            --json databaseId,status,createdAt \
            --jq '.[] | select(.status=="completed") | [.databaseId,.createdAt] | @tsv'
    )
    return 1
}

collect_expected_names() {
    local run_id="$1"
    local name
    while IFS= read -r name; do
        if should_ignore "$name"; then
            continue
        fi
        printf '%s\n' "$name"
    done < <(
        gh run view "$run_id" --repo "$GH_REPO" --json jobs --jq '.jobs[].name'
    )
}

job_status() {
    local jobs_json="$1"
    local name="$2"
    jq -r --arg n "$name" '
        [.jobs[] | select(.name == $n) | .status] as $s
        | if ($s | length) == 0 then
            "missing"
          elif ($s | map(. == "completed") | all) then
            "completed"
          else
            ($s | map(select(. != "completed")) | first)
          end
    ' <<<"$jobs_json"
}

started="$(date -u +%s)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
expected_file="${tmp_dir}/expected"
: >"$expected_file"

push_run="$(pick_reference_run "On Push" || true)"
schedule_run="$(pick_reference_run "On Schedule" || true)"

if [ -n "$push_run" ]; then
    collect_expected_names "$push_run" >>"$expected_file"
fi
if [ -n "$schedule_run" ]; then
    collect_expected_names "$schedule_run" >>"$expected_file"
fi

sort -u -o "$expected_file" "$expected_file"

filtered_file="${tmp_dir}/filtered"
while IFS= read -r name; do
    # Skip unexpanded reusable-workflow placeholders when nested jobs exist
    # (e.g. "CI / GithubPage" vs "CI / GithubPage / Extract Version").
    if grep -F -q "${name} / " "$expected_file"; then
        echo "dropping placeholder job '${name}'"
        continue
    fi
    printf '%s\n' "$name"
done <"$expected_file" >"$filtered_file"
mv "$filtered_file" "$expected_file"
expected_count="$(wc -l <"$expected_file" | tr -d ' ')"

if [ "$expected_count" -eq 0 ]; then
    echo "Could not build the On Push / On Schedule job set from main." >&2
    exit 1
fi

echo "Waiting for ${expected_count} On Push / On Schedule jobs in run ${GITHUB_RUN_ID}"
echo "Expected jobs:"
cat "$expected_file"

while true; do
    now="$(date -u +%s)"
    elapsed="$((now - started))"
    jobs_json="$(gh run view "$GITHUB_RUN_ID" --repo "$GH_REPO" --json jobs)"

    pending=0
    pending_report="${tmp_dir}/pending"
    : >"$pending_report"
    while IFS= read -r name; do
        status="$(job_status "$jobs_json" "$name")"
        case "$status" in
            completed) ;;
            queued | in_progress | waiting | pending | requested | waiting_for_progress | missing)
                echo "${name}: ${status}" >>"$pending_report"
                pending=1
                ;;
            *)
                echo "${name}: ${status}" >>"$pending_report"
                pending=1
                ;;
        esac
    done <"$expected_file"

    pending_count="$(wc -l <"$pending_report" | tr -d ' ')"
    echo "elapsed=${elapsed}s expected=${expected_count} not_done=${pending_count}"

    if [ "$elapsed" -gt "$TIMEOUT_SECONDS" ]; then
        echo "Timed out waiting for On Push / On Schedule jobs:" >&2
        cat "$pending_report" >&2
        exit 1
    fi

    if [ "$pending" -eq 0 ]; then
        echo "All On Push / On Schedule jobs have completed."
        exit 0
    fi

    echo "Still queued / in_progress / missing:"
    head -n 20 "$pending_report"
    if [ "$pending_count" -gt 20 ]; then
        echo "... ($((pending_count - 20)) more)"
    fi
    rm -f "$pending_report"

    sleep "$POLL_SECONDS"
done
