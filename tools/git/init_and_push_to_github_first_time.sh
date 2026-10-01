#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <github-repository-url>"
    echo "Example:"
    echo "  $0 https://github.com/BeppeMagro/pymkm.git"
    exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "This directory is already inside a Git repository. Aborting."
    exit 1
fi

REMOTE_URL="$1"

echo "[1/4] Initializing Git repository..."
git init

echo "[2/4] Adding files..."
git add -A

echo "[3/4] Creating initial commit..."
git commit -m "Initial commit"

echo "[4/4] Configuring origin and pushing main..."
git branch -M main
git remote add origin "$REMOTE_URL"
git push -u origin main

echo "Repository initialized and pushed successfully."