# Contributing

Thanks for your interest in contributing to this project! This file gives a short checklist and instructions to make contributions easy to review and merge.

## Quick Start
- Fork the repository and create a feature branch from `main`:
  ```bash
  git checkout -b feature/your-short-description
  ```
- Make changes on your branch and keep commits small and focused.
- Open a Pull Request (PR) against `main` with a clear description and motivation.

## Development Setup
1. Clone and create virtual environment:
   ```bash
   git clone https://github.com/muk0644/autonomous-agent-q-learning-dqn.git
   cd autonomous-agent-q-learning-dqn
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Recommended editor: VS Code (repository includes `.vscode/settings.json` to help Pylance find local modules).
3. Recommended extensions: Python, Pylance, Jupyter.

## Code Style & Tests
- Use clear, descriptive names and keep functions small.
- Follow PEP8 for Python. Optionally use `black` or `flake8`.
- If you add code, include simple tests or a short example demonstrating behavior.

## Commit Messages
- Prefer short, structured messages. Example (Conventional Commits):
  - `feat: add training script for DQN`
  - `fix: resolve padm_env import path`
  - `docs: update README with Docker instructions`

## Pull Request Checklist
- [ ] The branch is up to date with `main` (rebase or merge as appropriate)
- [ ] Code is linted and formatted
- [ ] Added or updated documentation if needed
- [ ] Small, focused commits with descriptive messages

## Issues
- When opening an issue, try to include:
  - A short descriptive title
  - Steps to reproduce (if bug)
  - Expected vs actual behavior
  - Environment details (OS, Python version)

## Notebooks
- `tutorial.ipynb` is an interactive tutorial. Large output cells are best cleared before committing.
- Prefer adding code changes as `.py` modules when possible; keep notebooks for demonstration.

## VS Code Settings
- `.vscode/settings.json` is included intentionally to help Pylance resolve local imports (e.g., `padm_env`). This speeds up onboarding for contributors using VS Code. If you prefer not to use it, you can safely ignore or remove it locally.

## Contact
If you have questions or want help, open an issue or contact the repository owner: `muk0644` (see README for email/contact links).

---
Thank you for helping improve this project! Your contributions make this repo better for everyone.