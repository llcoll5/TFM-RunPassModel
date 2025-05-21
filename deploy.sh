#!/bin/sh
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please provide a commit message."
    exit 1
fi
echo "Adding all files to git..."
git add *
echo "Committing changes..."
git commit -m "$1"
echo "pushing to GitHub..."
git push
echo "Deployed to GitHub Pages"