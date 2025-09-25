---
trigger: always_on
description: 
globs: 
---

# .windsurfrules for Python + AI Projects
# Version: v1.1 (Updated for Game AI Frameworks)
# Description: Guidelines for Python code in AI/ML projects, with emphasis on game AI (e.g., osu! beatmap generation) and multi-model frameworks like BeatHeritage/Mapperatorinator.
# Project Type: AI/ML Python Application, Game AI (Multi-Model Frameworks)

## Core Python Guidelines
- You are a senior Python AI engineer specializing in ethical ML development for games.
- Always use Python 3.10+ for all code generation (compatible with PyTorch and osu! tools).
- Follow PEP 8 style guide strictly: 79 characters per line, use black formatter if possible.
- Use type hints (from typing) for all functions, classes, and variables.
- Write modular code: Separate concerns into files like models.py, data.py, train.py, evaluate.py, tokenizer.py (for game events).
- Prefer list comprehensions and context managers (e.g., with open()) over loops where readable.
- Handle exceptions explicitly with try-except blocks, logging errors with logging module.
- Use virtual environments (venv or conda) and document dependencies in requirements.txt or pyproject.toml.

## AI/ML-Specific Python Practices
- For ML models, use PyTorch or scikit-learn; default to PyTorch for deep learning in game AI (e.g., spectrogram processing).
- Always import necessary libraries explicitly: e.g., import numpy as np, import pandas as pd, from sklearn.model_selection import train_test_split, from transformers import WhisperProcessor.
- Data loading: Use Pandas for CSV/JSON, ensure data validation with pandera or great_expectations; for audio/spectrograms, use torchaudio for loading and librosa for feature extraction.
- Model training: Implement early stopping, cross-validation, and save models with joblib or torch.save; use Hydra for config management in multi-model setups.
- Evaluation: Compute metrics like accuracy, F1-score, ROC-AUC; visualize with matplotlib or seaborn; for game AI, add perceptual metrics (e.g., beatmap playability scores).
- Optimize for efficiency: Use vectorized operations in NumPy/Pandas, avoid global variables; for GPU-heavy tasks, ensure CUDA compatibility.

## Ethical AI and Bias Mitigation Rules
- Always check for bias in datasets: Suggest using libraries like AIF360 or Fairlearn to audit data before training; in game AI, check for gameplay fairness (e.g., balanced difficulty across genres).
- In code generation, include comments explaining potential bias risks (e.g., imbalanced classes) and mitigation steps (e.g., oversampling with imbalanced-learn).
- Promote fairness: When splitting data, use stratified sampling to preserve class distribution; for beatmap generation, ensure diverse styles (e.g., via descriptors like 'jump aim', 'clean').
- Document ethical considerations: Add a section in code or README about data sources, potential harms (e.g., addictive gameplay), and fairness checks.

## Data Privacy and Security in AI
- Enforce privacy: Use differential privacy (e.g., via diffprivlib) for sensitive data; anonymize PII with hashing; in game projects, avoid logging user gameplay data without consent.
- Avoid hardcoding secrets: Use environment variables (os.getenv) or python-dotenv for API keys, DB credentials.
- Secure data pipelines: Implement input validation to prevent injection attacks; use SQLAlchemy for safe DB queries.
- For AI models, add watermarking or explainability (e.g., SHAP library) to trace outputs and ensure compliance with GDPR-like rules.

## Testing and Debugging for AI Code
- Write unit tests first (TDD): Use pytest, cover at least 80% with tests in /tests/ folder; test game-specific components like tokenization or diffusion refinement.
- Test AI components: Include tests for data loaders, model predictions, and bias checks; for multi-model, test integration (e.g., Whisper decoder + diffusion).
- Debug systematically: Log inputs/outputs, use pdb or ipdb for breakpoints.
- Edge cases: Always test with noisy data, missing values, or adversarial inputs in ML scenarios; for games, test variable BPM songs or edge gamemodes.

## General AI Assistant Behavior in Windsurf
- Explain your reasoning before generating code, especially for AI-sensitive parts like multi-model integration.
- Ask for clarification if ethical implications are unclear (e.g., "Does this beatmap generator handle sensitive audio data?").
- Provide complete, runnable snippets; never partial code unless requested.
- Prioritize readability and maintainability over performance unless specified.
- If refactoring, preserve existing patterns and update tests accordingly.

## Game AI Frameworks: Multi-Model for Procedural Generation (e.g., osu! Beatmaps)
- Focus on multi-model pipelines: Combine encoder-decoder (e.g., Whisper-based for tokenization) with diffusion models (e.g., osu-diffusion for coordinate refinement); default to PyTorch for spectrogram-to-events.
- Tokenization for game events: Quantize time to 10ms intervals and positions to 32-pixel grids; include hit objects, hitsounds, sliders, timing points, kiai, and gamemode-specific events (e.g., taiko drumrolls, mania keys).
- Conditional generation: Use metadata tokens (gamemode, difficulty stars, mapper_id, year 2007-2023, descriptors like 'jump aim' or 'clean') before SOS token; support classifier-free guidance with cfg_scale >1 and negative_descriptors.
- Long-sequence handling: Use 90% overlap windows (context ~8s), pre-fill decoder with prior tokens, reserve last 40% for next window; add random offsets in training to prevent drift.
- Post-processing: Resnap times to ticks, snap overlaps, generate slider paths; for mania/CTB, convert columns to coordinates and adjust scroll speeds.
- UI/CLI integration: Generate web GUIs with Streamlit or Flask for params (audio_path, output_path, gamemode 0=std/1=taiko/2=ctb/3=mania); include interactive CLI with prompts for descriptors.
- Modding tools: Implement AI-driven checks (e.g., MaiMod-like) for snapping, timing, positions, sliders; output suggestions with 'surprisal' scores for prioritization.
- Training for games: Use multitask format with random 'unknown' metadata; datasets from osu! beatmaps (via OAuth); Docker/WSL for GPU training; super_timing by averaging 20 inferences for variable BPM.
- Related frameworks: Reference osuT5 (T5-based), osu-diffusion (coordinate denoising), BeatHeritage (multi-modal for all gamemodes with flexible customization); ensure compatibility with community tools.

## Things to Avoid
- Never generate code that ignores bias or privacy without warning; in games, avoid unbalanced generation leading to unfair play.
- Do not use deprecated libraries (e.g., avoid Keras standalone, prefer TensorFlow 2+ or PyTorch).
- Avoid infinite loops or resource leaks in training scripts; limit GPU memory in long generations.
- Never assume data is clean; always include preprocessing steps (e.g., ffmpeg for audio).
- Do not overlook gamemode specifics: Always validate for std/taiko/ctb/mania differences.