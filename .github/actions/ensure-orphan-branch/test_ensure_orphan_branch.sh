#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${ROOT}/ensure-orphan-branch.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_eq() {
    local left="$1"
    local right="$2"
    local msg="$3"
    [[ "${left}" == "${right}" ]] || fail "${msg}: expected '${right}', got '${left}'"
}

assert_file_contains() {
    local file="$1"
    local needle="$2"
    grep -qx "${needle}" "${file}" || fail "${file} missing '${needle}'"
}

commit_and_push() {
    local dir="$1"
    local branch="$2"
    local file="$3"
    local content="$4"
    git init -b "${branch}" "${dir}" >/dev/null
    git -C "${dir}" config user.name test
    git -C "${dir}" config user.email test@example.com
    printf '%s\n' "${content}" >"${dir}/${file}"
    git -C "${dir}" add "${file}"
    git -C "${dir}" commit -m "${branch}" >/dev/null
    git -C "${dir}" push "${BARE}" "HEAD:refs/heads/${branch}" >/dev/null
}

BARE="${TMP}/origin.git"
git init --bare -b main "${BARE}" >/dev/null

# Populate origin/main so we can prove the orphan branch does not copy it.
commit_and_push "${TMP}/seed" main tracked-on-main.txt from-main

# A similarly named branch must not be treated as a match (the original
# workflow grepped ls-remote output as a substring).
commit_and_push "${TMP}/extra" orphan-branch-extra README.md "# extra"

export BRANCH_NAME=orphan-branch
export REMOTE_URL="${BARE}"
export COMMIT_MESSAGE="Initial commit on orphan branch"
export GIT_USER_NAME=test
export GIT_USER_EMAIL=test@example.com
unset GITHUB_TOKEN || true

OUT1="${TMP}/out1.txt"
GITHUB_OUTPUT="${OUT1}" "${SCRIPT}"

assert_file_contains "${OUT1}" "branch_name=orphan-branch"
assert_file_contains "${OUT1}" "exists=false"
assert_file_contains "${OUT1}" "created=true"

git --git-dir="${BARE}" rev-parse --verify refs/heads/orphan-branch >/dev/null \
    || fail "orphan-branch was not created"

COMMIT="$(git --git-dir="${BARE}" rev-parse refs/heads/orphan-branch)"
PARENTS="$(git --git-dir="${BARE}" rev-list --parents -n 1 "${COMMIT}" | awk '{print NF}')"
# rev-list --parents prints "<commit> [parent...]", so a root commit has one field.
assert_eq "${PARENTS}" "1" "orphan commit should have no parents"

TREE="$(git --git-dir="${BARE}" ls-tree --name-only "${COMMIT}")"
assert_eq "${TREE}" "README.md" "orphan tree should contain only README.md"

CONTENT="$(git --git-dir="${BARE}" show "${COMMIT}:README.md")"
assert_eq "${CONTENT}" "# orphan-branch" "README.md content"

git --git-dir="${BARE}" cat-file -e "refs/heads/orphan-branch:tracked-on-main.txt" 2>/dev/null \
    && fail "orphan branch must not contain files from main"

OUT2="${TMP}/out2.txt"
GITHUB_OUTPUT="${OUT2}" "${SCRIPT}"
assert_file_contains "${OUT2}" "exists=true"
assert_file_contains "${OUT2}" "created=false"

COMMIT2="$(git --git-dir="${BARE}" rev-parse refs/heads/orphan-branch)"
assert_eq "${COMMIT2}" "${COMMIT}" "second run must not rewrite the branch"

# Invalid ref names must fail.
if BRANCH_NAME='bad branch' GITHUB_OUTPUT="${TMP}/out-bad.txt" "${SCRIPT}"; then
    fail "invalid branch name should be rejected"
fi

# Custom branch name.
export BRANCH_NAME=docs/empty-root
OUT3="${TMP}/out3.txt"
GITHUB_OUTPUT="${OUT3}" "${SCRIPT}"
assert_file_contains "${OUT3}" "created=true"
git --git-dir="${BARE}" rev-parse --verify refs/heads/docs/empty-root >/dev/null \
    || fail "docs/empty-root was not created"

echo "All ensure-orphan-branch tests passed."
