#!/bin/bash

# Delete Git tags locally and remotely.
# Usage:
#   ./tools/git/delete_tags.sh v1.0.0 v1.0.1

if [ $# -eq 0 ]; then
  echo "No tags specified."
  echo "Usage: $0 <tag1> <tag2> ..."
  exit 1
fi

for TAG in "$@"; do
  echo "Deleting tag $TAG..."

  if git rev-parse "$TAG" >/dev/null 2>&1; then
    git tag -d "$TAG"
    echo "Deleted local tag $TAG"
  else
    echo "Local tag $TAG not found"
  fi

  if git ls-remote --tags origin | grep -q "refs/tags/$TAG"; then
    git push --delete origin "$TAG"
    echo "Deleted remote tag $TAG"
  else
    echo "Remote tag $TAG not found on origin"
  fi
done