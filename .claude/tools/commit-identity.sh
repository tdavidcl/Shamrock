#!/bin/bash
# Apply AGENTS.md's "Commit authorship" rules to a commit made by an agent.
#
# The author is the human who started the session: their email comes from
# $CLAUDE_CODE_USER_EMAIL (set by Claude Code to the account's email) and is
# resolved through the repo's .mailmap to the canonical "Name <email>". The
# message keeps a single "Assisted-by: <agent>" trailer and loses the lines
# agent tooling injects (Co-authored-by, Claude-Session, session URLs).
#
# Usage: .claude/tools/commit-identity.sh [--email <email>] [--agent <name>] <command>
#   author   print the mailmap-resolved author, e.g. for `git commit --author`
#   trailer  print the Assisted-by trailer line
#   amend    rewrite HEAD's author and message in place (with --no-verify,
#            so the commit-msg hook does not inject its lines again)
set -euo pipefail

email="${CLAUDE_CODE_USER_EMAIL:-}"
agent="Claude Code"

die() {
  echo "commit-identity: $*" >&2
  exit 1
}

usage() {
  sed -n '/^# Usage:/,/^set /{/^set /d;s/^# \{0,1\}//;p}' "$0"
  exit "${1:-0}"
}

resolve_author() {
  [ -n "$email" ] || die "no email: \$CLAUDE_CODE_USER_EMAIL is unset, pass --email"
  local mapped
  mapped=$(git check-mailmap "<$email>")
  # An address with no .mailmap entry comes back as a bare "<email>".
  case "$mapped" in
    "<"*) die "no .mailmap entry gives a name for <$email>; add one to .mailmap" ;;
  esac
  echo "$mapped"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --email) email="${2:?--email needs a value}"; shift 2 ;;
    --agent) agent="${2:?--agent needs a value}"; shift 2 ;;
    -h|--help) usage ;;
    -*) die "unknown option $1" ;;
    *) break ;;
  esac
done

case "${1:-}" in
  author)
    resolve_author
    ;;
  trailer)
    echo "Assisted-by: $agent"
    ;;
  amend)
    author=$(resolve_author)
    msg=$(mktemp)
    trap 'rm -f "$msg"' EXIT
    git log -1 --format=%B HEAD \
      | { grep -viE '^(co-authored-by|claude-session):|claude\.ai/code/session_' || true; } \
      | git interpret-trailers --if-exists replace --trailer "Assisted-by: $agent" \
      > "$msg"
    git commit --quiet --amend --no-verify --cleanup=whitespace \
      --author="$author" -F "$msg"
    git log -1 --format='Author: %an <%ae>%n%n%B'
    ;;
  *)
    usage 1 >&2
    ;;
esac
