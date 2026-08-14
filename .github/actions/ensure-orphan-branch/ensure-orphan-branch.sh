#!/usr/bin/env bash
# Ensure a named orphan branch exists on a remote.
# Used by the ensure-orphan-branch composite action; also runnable locally.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

write_output() {
    local key="$1"
    local value="$2"
    echo "${key}=${value}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "${key}=${value}" >>"${GITHUB_OUTPUT}"
    fi
}

die() {
    echo "::error::$*"
    exit 1
}

run_git() {
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        local basic
        basic="$(printf 'x-access-token:%s' "${GITHUB_TOKEN}" | openssl base64 -A)"
        git -c "http.extraheader=AUTHORIZATION: basic ${basic}" "$@"
    else
        git "$@"
    fi
}

branch_exists() {
    local listing sha ref
    listing="$(run_git ls-remote --heads "${REMOTE_URL}" "refs/heads/${BRANCH_NAME}")" \
        || die "git ls-remote failed for ${REMOTE_URL}"
    while read -r sha ref; do
        [[ -n "${ref}" ]] || continue
        if [[ "${ref}" == "refs/heads/${BRANCH_NAME}" ]]; then
            return 0
        fi
    done <<<"${listing}"
    return 1
}

BRANCH_NAME="${BRANCH_NAME:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Initial commit on orphan branch}"
GIT_USER_NAME="${GIT_USER_NAME:-github-actions[bot]}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-41898282+github-actions[bot]@users.noreply.github.com}"

if [[ -z "${REMOTE_URL:-}" ]]; then
    if [[ -z "${REPOSITORY:-}" || -z "${GITHUB_SERVER_URL:-}" ]]; then
        die "Set REMOTE_URL, or REPOSITORY and GITHUB_SERVER_URL."
    fi
    REMOTE_URL="${GITHUB_SERVER_URL%/}/${REPOSITORY}.git"
fi

[[ -n "${BRANCH_NAME}" ]] || die "BRANCH_NAME is required."

git check-ref-format "refs/heads/${BRANCH_NAME}" \
    || die "Invalid branch name '${BRANCH_NAME}'."

write_output "branch_name" "${BRANCH_NAME}"

echo "Checking whether branch '${BRANCH_NAME}' exists on ${REMOTE_URL}"

if branch_exists; then
    echo "Branch '${BRANCH_NAME}' already exists."
    write_output "exists" "true"
    write_output "created" "false"
    exit 0
fi

echo "Branch '${BRANCH_NAME}' is missing; creating an orphan branch."

if [[ "${REMOTE_URL}" == https://* && -z "${GITHUB_TOKEN:-}" ]]; then
    die "github_token is required to create '${BRANCH_NAME}' on ${REMOTE_URL}."
fi

WORKDIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORKDIR}"
}
trap cleanup EXIT

git init -b "${BRANCH_NAME}" "${WORKDIR}"
git -C "${WORKDIR}" config user.name "${GIT_USER_NAME}"
git -C "${WORKDIR}" config user.email "${GIT_USER_EMAIL}"

printf '# %s\n' "${BRANCH_NAME}" >"${WORKDIR}/README.md"
git -C "${WORKDIR}" add README.md
git -C "${WORKDIR}" commit -m "${COMMIT_MESSAGE}"

if run_git -C "${WORKDIR}" push "${REMOTE_URL}" "HEAD:refs/heads/${BRANCH_NAME}"; then
    echo "Created orphan branch '${BRANCH_NAME}'."
    write_output "exists" "false"
    write_output "created" "true"
    exit 0
fi

if branch_exists; then
    echo "Branch '${BRANCH_NAME}' was created concurrently; nothing to do."
    write_output "exists" "true"
    write_output "created" "false"
    exit 0
fi

die "Failed to create orphan branch '${BRANCH_NAME}'."
