# BeatHeritage V1 Model Documentation

## Overview
BeatHeritage V1 is an enhanced version of the Mapperatorinator V30 model, designed specifically for improved stability and generation quality in osu! beatmap creation. This model represents a significant advancement in AI-driven beatmap generation with enhanced features and optimizations.

## Model Information
- **Model Path:** `hongminh54/BeatHeritage-v1`
- **Base Architecture:** Whisper-based model (219M parameters)
- **Version:** BeatHeritage V1
- **Release Date:** September 2025

## Configuration Files
- **Inference Config:** `configs/inference/beatheritage_v1.yaml`
- **Training Config:** `configs/train/beatheritage_v1.yaml`
- **Diffusion Checkpoint:** `hongminh54/osu-diffusion-v2`

## Key Improvements Over Mapperatorinator V30

### 1. Enhanced Sampling Parameters
- **Temperature:** 0.85 (reduced from 0.9) for better stability
- **Top-p:** 0.92 (increased from 0.9) for improved diversity
- **Top-k:** 50 (newly added) for better control
- **Repetition Penalty:** 1.1 (new) to reduce repetitive patterns

### 2. Quality Control Features
- **Min Distance Threshold:** 20 pixels between objects
- **Max Overlap Ratio:** 0.15 maximum allowed overlap
- **Auto-correction:** Automatic spacing issue fixes
- **Flow Optimization:** Enhanced flow pattern generation

### 3. Advanced Generation Features
- **Context-Aware Generation:** Better understanding of beatmap context
- **Style Preservation:** Maintains consistent mapping style
- **Difficulty Scaling:** Improved difficulty progression
- **Pattern Variety:** More diverse pattern generation

### 4. Training Enhancements
- **Flash Attention:** Enabled for better performance
- **Dataset Size:** 40,000 training samples (expanded from 38,689)
- **Gamemode Support:** All gamemodes (0, 1, 2, 3)
- **Data Augmentation:** Rotation, flip, scale, and noise
- **Regularization:** Weight decay (0.01) and gradient clipping (1.0)

### 5. Performance Optimizations
- **Batch Processing:** Optimized batch size (48 with gradient accumulation)
- **Mixed Precision:** BF16 precision for better stability
- **Caching:** 4096 context cache size
- **Memory Efficiency:** Gradient checkpointing enabled

## Usage

### Web UI
Select "BeatHeritage V1 (Enhanced stability & quality)" from the model dropdown in the web interface.

### CLI
```bash
# When prompted for model selection, choose option 5
Select Model:
  1) Mapperatorinator V28
  2) Mapperatorinator V29 (Supports gamemodes and descriptors)
  3) Mapperatorinator V30 (Best stable model)
  4) Mapperatorinator V31 (Slightly more accurate than V29)
  5) BeatHeritage V1 (Enhanced stability & quality)
```

### Python API
```python
python inference.py -cn beatheritage_v1 \
    audio_path='path/to/audio.mp3' \
    output_path='output/' \
    gamemode=0 \
    difficulty=5.5
```

## Model Features

### Supported Tokens
- **Gamemode tokens:** std, taiko, ctb, mania
- **Difficulty tokens:** 1.0-10.0 star rating
- **Style tokens:** jump aim, stream, tech, flow, clean, complex
- **Mapper ID:** Style-specific generation
- **Year tokens:** 2007-2023
- **Special tokens:** timing, kiai, hitsounds, SV

### Context Types
- Map context
- Timing context
- Map-to-map learning

### Post-processing
- Automatic resnapping to ticks
- Overlap detection and fixing
- Slider path generation
- Coordinate refinement via diffusion

## Best Practices

### For Best Results
1. Use high-quality audio files (MP3 or OGG)
2. Specify appropriate difficulty rating
3. Use descriptors for style guidance
4. Enable super_timing for variable BPM songs
5. Use in-context learning with reference beatmaps

### Common Settings
```yaml
temperature: 0.85
top_p: 0.92
cfg_scale: 7.5
generate_positions: true
position_refinement: true
```

## Limitations
- Maximum context length: 8.192 seconds
- Requires CUDA-compatible GPU for optimal performance
- Best results with songs under 5 minutes

## Troubleshooting

### If generation quality is poor:
1. Lower temperature to 0.7-0.8
2. Increase cfg_scale to 10-15
3. Use more specific descriptors
4. Provide reference beatmap for context

### If generation is too repetitive:
1. Increase repetition_penalty to 1.2-1.5
2. Increase top_p to 0.95
3. Use negative descriptors to avoid patterns

## Future Improvements
- Extended context length support
- Real-time generation capabilities
- Multi-difficulty set generation
- Enhanced gamemode-specific features

## Credits
- Based on Mapperatorinator by OliBomby
- Enhanced by hongminh54
- Using osu-diffusion for coordinate refinement

## License
Same as the original BeatHeritage/Mapperatorinator project

---

For more information, see the main [README.md](../README.md) or visit the project repository.
