# 🚀 GitHub Deployment Commands

## After Installing Git, run these commands in order:

### 1. Navigate to Project Directory
```bash
cd "F:\RVR AI CAPTCHA SOLVER"
```

### 2. Initialize Git Repository
```bash
git init
```

### 3. Configure Git (Replace with your info)
```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 4. Add Remote Repository
```bash
git remote add origin https://github.com/khiasu/AI-Reverse-CAPTCHA-Solver.git
```

### 5. Add All Files
```bash
git add .
```

### 6. Make Initial Commit
```bash
git commit -m "Initial commit: AI CAPTCHA Solver with CNN model, preprocessing, inference, OCR comparison, and Streamlit demo"
```

### 7. Push to GitHub
```bash
git push -u origin main
```

## If you get authentication errors:
```bash
# Use GitHub CLI (recommended)
gh auth login

# Or use personal access token instead of password
# Go to GitHub Settings > Developer settings > Personal access tokens
# Generate new token with repo permissions
```

## Verify Deployment
After pushing, visit your GitHub repository to confirm all files are uploaded correctly.

## Next Steps After Deployment
1. Add repository description and topics on GitHub
2. Create releases for major versions
3. Set up GitHub Actions for CI/CD (optional)
4. Deploy demo to cloud platforms (Streamlit Cloud, Heroku, etc.)
