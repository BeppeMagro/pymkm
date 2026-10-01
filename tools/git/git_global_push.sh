#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 \"commit message\""
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

BRANCH=$(git branch --show-current)

if [ "$BRANCH" != "main" ]; then
    echo "Current branch is '$BRANCH', not 'main'. Aborting."
    exit 1
fi

git add -A

if git diff --cached --quiet; then
    echo "No changes to commit."
    exit 0
fi

git commit -m "$1"
git push origin main