from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from osuT5.osuT5.config import TrainConfig
from osuT5.osuT5.tokenizer import ContextType
from osu_diffusion.config import DiffusionTrainConfig


# BeatHeritage V1 Config Sections

@dataclass
class AdvancedFeaturesConfig:
    enable_context_aware_generation: bool = True
    enable_style_preservation: bool = True
    enable_difficulty_scaling: bool = True
    enable_pattern_variety: bool = True

@dataclass
class QualityControlConfig:
    min_distance_threshold: int = 20
    max_overlap_ratio: float = 0.15
    enable_auto_correction: bool = True
    enable_flow_optimization: bool = True

@dataclass
class PerformanceConfig:
    use_flash_attention: bool = False
    batch_size: int = 1
    max_sequence_length: int = 5120
    cache_size: int = 4096

@dataclass
class MetadataConfig:
    preserve_timing_points: bool = True
    preserve_bookmarks: bool = True
    auto_detect_kiai: bool = True
    smart_hitsounding: bool = True

@dataclass
class PostprocessorConfig:
    use_custom: bool = True
    class_name: str = 'beatheritage_postprocessor.BeatHeritagePostprocessor'
    config_class: str = 'beatheritage_postprocessor.BeatHeritageConfig'

@dataclass
class IntegrationsConfig:
    mai_mod_enhanced: bool = True
    fid_evaluation: bool = True
    benchmark_mode: bool = False


# Default config here based on V28

@dataclass
class InferenceConfig:
    model_path: str = ''  # Path to trained model
    audio_path: str = ''  # Path to input audio
    output_path: str = ''  # Path to output directory
    beatmap_path: str = ''  # Path to .osu file to autofill metadata and use as reference

    # Conditional generation settings
    gamemode: Optional[int] = None  # Gamemode of the beatmap
    beatmap_id: Optional[int] = None  # Beatmap ID to use as style
    difficulty: Optional[float] = None  # Difficulty star rating to map
    mapper_id: Optional[int] = None  # Mapper ID to use as style
    year: Optional[int] = None  # Year to use as style
    hitsounded: Optional[bool] = None  # Whether the beatmap has hitsounds
    keycount: Optional[int] = None  # Number of keys to use for mania
    hold_note_ratio: Optional[float] = None  # Ratio of how many hold notes to generate in mania
    scroll_speed_ratio: Optional[float] = None  # Ratio of how many scroll speed changes to generate in mania and taiko
    descriptors: Optional[list[str]] = None  # List of descriptors to use for style
    negative_descriptors: Optional[list[str]] = None  # List of descriptors to avoid when using classifier-free guidance

    # Difficulty settings
    hp_drain_rate: Optional[float] = None  # HP drain rate (HP)
    circle_size: Optional[float] = None  # Circle size (CS)
    overall_difficulty: Optional[float] = None  # Overall difficulty (OD)
    approach_rate: Optional[float] = None  # Approach rate (AR)
    slider_multiplier: Optional[float] = None  # Multiplier for slider velocity
    slider_tick_rate: Optional[float] = None  # Rate of slider ticks

    # Inference settings
    seed: Optional[int] = None  # Random seed
    device: str = 'auto'  # Inference device (cpu/cuda/mps/auto)
    precision: str = 'fp32'         # Lower precision for speed (fp32/bf16/amp)
    add_to_beatmap: bool = False  # Add generated content to the reference beatmap
    export_osz: bool = False  # Export beatmap as .osz file
    start_time: Optional[int] = None  # Start time of audio to generate beatmap for
    end_time: Optional[int] = None  # End time of audio to generate beatmap for
    lookback: float = 0.5  # Fraction of audio sequence to fill with tokens from previous inference window
    lookahead: float = 0.4  # Fraction of audio sequence to skip at the end of the audio window
    timing_leniency: int = 20  # Number of milliseconds of error to allow for timing generation
    in_context: list[ContextType] = field(default_factory=lambda: [ContextType.NONE])  # Context types of other beatmap(s)
    output_type: list[ContextType] = field(default_factory=lambda: [ContextType.MAP])  # Output type (map, timing)
    cfg_scale: float = 1.0  # Scale of classifier-free guidance
    temperature: float = 1.0  # Sampling temperature
    timing_temperature: float = 0.1  # Sampling temperature for timing
    mania_column_temperature: float = 0.5  # Sampling temperature for mania columns
    taiko_hit_temperature: float = 0.5  # Sampling temperature for taiko hit types
    timeshift_bias: float = 0.0  # Logit bias for sampling timeshift tokens
    top_p: float = 0.95  # Top-p sampling threshold
    top_k: int = 0  # Top-k sampling threshold
    repetition_penalty: float = 1.0  # Repetition penalty to reduce repetitive patterns
    parallel: bool = False  # Use parallel sampling
    do_sample: bool = True  # Use sampling
    num_beams: int = 1  # Number of beams for beam search
    super_timing: bool = False  # Use super timing generator (slow but accurate timing)
    timer_num_beams: int = 2  # Number of beams for beam search
    timer_bpm_threshold: float = 0.7  # Threshold requirement for BPM change in timer, higher values will result in less BPM changes
    timer_cfg_scale: float = 1.0  # Scale of classifier-free guidance for timer
    timer_iterations: int = 20  # Number of iterations for timer
    use_server: bool = True  # Use server for optimized multiprocess inference
    max_batch_size: int = 16  # Maximum batch size for inference (only used for parallel sampling or super timing)
    resnap_events: bool = True  # Resnap notes to the timing after generation
    position_refinement: bool = False  # Use position refinement

    # Metadata settings
    bpm: int = 120  # Beats per minute of input audio
    offset: int = 0  # Start of beat, in miliseconds, from the beginning of input audio
    title: str = ''  # Song title
    artist: str = ''  # Song artist
    creator: str = ''  # Beatmap creator
    version: str = ''  # Beatmap version
    background: Optional[str] = None  # File name of background image
    preview_time: int = -1  # Time in milliseconds to start previewing the song

    # Diffusion settings
    generate_positions: bool = True  # Use diffusion to generate object positions
    diff_cfg_scale: float = 1.0  # Scale of classifier-free guidance
    compile: bool = False  # PyTorch 2.0 optimization
    pad_sequence: bool = False  # Pad sequence to max_seq_len
    diff_ckpt: str = ''  # Path to checkpoint for diffusion model
    diff_refine_ckpt: str = ''  # Path to checkpoint for refining diffusion model
    beatmap_idx: str = 'osu_diffusion/beatmap_idx.pickle'  # Path to beatmap index
    refine_iters: int = 10  # Number of refinement iterations
    random_init: bool = False  # Whether to initialize with random noise instead of positions generated by the previous model
    timesteps: list[int] = field(default_factory=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # The number of timesteps we want to take from equally-sized portions of the original process
    max_seq_len: int = 1024  # Maximum sequence length for diffusion
    overlap_buffer: int = 128  # Buffer zone at start and end of sequence to avoid edge effects (should be less than half of max_seq_len)

    # Training settings
    train: TrainConfig = field(default_factory=TrainConfig)  # Training settings for osuT5 model
    diffusion: DiffusionTrainConfig = field(default_factory=DiffusionTrainConfig)  # Training settings for diffusion model

    # BeatHeritage V1 Config Sections
    advanced_features: AdvancedFeaturesConfig = field(default_factory=AdvancedFeaturesConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    postprocessor: PostprocessorConfig = field(default_factory=PostprocessorConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    hydra: Any = MISSING


@dataclass
class FidConfig:
    device: str = 'auto'  # Inference device (cpu/cuda/mps/auto)
    compile: bool = True
    num_processes: int = 3
    seed: int = 0

    skip_generation: bool = False
    fid: bool = True
    rhythm_stats: bool = True

    dataset_type: str = 'ors'
    dataset_path: str = '/workspace/datasets/ORS16291'
    dataset_start: int = 16200
    dataset_end: int = 16291
    gamemodes: list[int] = field(default_factory=lambda: [0])  # List of gamemodes to include in the dataset

    classifier_ckpt: str = 'OliBomby/osu-classifier'
    classifier_batch_size: int = 16

    training_set_ids_path: Optional[str] = None  # Path to training set beatmap IDs

    inference: InferenceConfig = field(default_factory=InferenceConfig)  # Training settings for osuT5 model
    hydra: Any = MISSING


@dataclass
class MaiModConfig:
    beatmap_path: str = ''  # Path to .osu file
    audio_path: str = ''  # Path to input audio
    raw_output: bool = False
    precision: str = 'fp32'         # Lower precision for speed (fp32/bf16/amp)
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # Training settings for osuT5 model
    hydra: Any = MISSING


cs = ConfigStore.instance()
cs.store(group="inference", name="base", node=InferenceConfig)
cs.store(name="base_fid", node=FidConfig)
cs.store(name="base_mai_mod", node=MaiModConfig)
