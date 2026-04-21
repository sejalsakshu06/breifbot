#!/bin/bash
# Run this script to push the project to GitHub for the first time
# Usage: bash push_to_github.sh

echo "🚀 AI Project Intelligence — GitHub Push Helper"
echo "------------------------------------------------"

read -p "Enter your GitHub username: " USERNAME
read -p "Enter your repo name (e.g. ai-project-intelligence): " REPO

echo ""
echo "Running the following commands:"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit: AI Project Intelligence System'"
echo "  git branch -M main"
echo "  git remote add origin https://github.com/$USERNAME/$REPO.git"
echo "  git push -u origin main"
echo ""
read -p "Proceed? (y/n): " CONFIRM

if [ "$CONFIRM" = "y" ]; then
    git init
    git add .
    git commit -m "Initial commit: AI Project Intelligence System

- RAG pipeline with FAISS + sentence-transformers
- NLP analysis: sentiment, keywords, key phrases, readability
- Groq LLM integration (free API) for generation
- Daily report generator (Markdown + PDF)
- Streamlit UI with multi-format file upload
- Unit tests with pytest"

    git branch -M main
    git remote add origin "https://github.com/$USERNAME/$REPO.git"
    git push -u origin main
    echo ""
    echo "✅ Done! Visit: https://github.com/$USERNAME/$REPO"
else
    echo "Aborted."
fi
