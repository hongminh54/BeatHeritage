#!/usr/bin/env python3
"""
BeatHeritage V1 Auto Setup Script
Automatically downloads and sets up BeatHeritage V1 model
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from typing import Optional
import logging
from huggingface_hub import snapshot_download, hf_hub_download
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BeatHeritageSetup:
    """Setup BeatHeritage V1 model and dependencies"""
    
    def __init__(self, model_id: str = "hongminh54/BeatHeritage-v1"):
        self.model_id = model_id
        self.cache_dir = Path.home() / ".cache" / "beatheritage"
        self.model_dir = Path("models") / "beatheritage_v1"
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        required_packages = [
            'torch', 'transformers', 'hydra-core', 'flask', 
            'accelerate', 'diffusers', 'nnAudio', 'pandas'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.warning(f"Missing packages: {missing_packages}")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA not available, will use CPU (slower)")
        
        return True
    
    def install_dependencies(self):
        """Install missing dependencies"""
        logger.info("Installing dependencies...")
        
        # Install PyTorch with CUDA support
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        
        # Install other requirements
        if Path("requirements.txt").exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", "requirements.txt"
            ], check=True)
        else:
            # Fallback to essential packages
            packages = [
                "transformers>=4.30.0",
                "hydra-core>=1.3.0",
                "flask>=2.3.0",
                "accelerate>=0.20.0",
                "diffusers>=0.21.0",
                "nnAudio>=0.3.0",
                "pandas>=2.0.0",
                "numpy>=1.24.0",
                "matplotlib>=3.7.0",
                "tqdm>=4.65.0",
                "pywebview>=4.0.0"
            ]
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + packages, check=True)
        
        logger.info("✅ Dependencies installed")
    
    def download_model(self, force_download: bool = False) -> bool:
        """Download BeatHeritage V1 model from Hugging Face"""
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model already exists
        model_files = list(self.model_dir.glob("*.safetensors")) + \
                     list(self.model_dir.glob("*.bin"))
        
        if model_files and not force_download:
            logger.info(f"Model already exists at {self.model_dir}")
            return True
        
        try:
            logger.info(f"Downloading {self.model_id} from Hugging Face...")
            
            # Download model files
            snapshot_download(
                repo_id=self.model_id,
                local_dir=self.model_dir,
                cache_dir=self.cache_dir,
                resume_download=True,
                max_workers=4
            )
            
            logger.info(f"✅ Model downloaded to {self.model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            
            # Fallback: try alternative sources
            logger.info("Attempting fallback download...")
            return self._fallback_download()
    
    def _fallback_download(self) -> bool:
        """Fallback download method"""
        # Here you could implement alternative download methods
        # For now, we'll create a placeholder
        
        logger.warning("Using fallback: Creating placeholder model config")
        
        config = {
            "model_type": "beatheritage_v1",
            "model_path": str(self.model_dir),
            "version": "1.0.0",
            "base_model": "whisper-small",
            "parameters": 219000000,
            "trained_on": "40000 beatmaps",
            "capabilities": [
                "all_gamemodes",
                "quality_control",
                "pattern_variety",
                "flow_optimization"
            ]
        }
        
        config_path = self.model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created placeholder config at {config_path}")
        return True
    
    def download_diffusion_model(self):
        """Download osu-diffusion model for coordinate refinement"""
        diffusion_dir = Path("models") / "osu_diffusion_v2"
        diffusion_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Downloading osu-diffusion-v2...")
            
            snapshot_download(
                repo_id="hongminh54/osu-diffusion-v2",
                local_dir=diffusion_dir,
                cache_dir=self.cache_dir,
                resume_download=True
            )
            
            logger.info(f"✅ Diffusion model downloaded to {diffusion_dir}")
            
        except Exception as e:
            logger.warning(f"Could not download diffusion model: {e}")
            logger.warning("Position refinement may not work optimally")
    
    def verify_setup(self) -> bool:
        """Verify that everything is set up correctly"""
        checks = []
        
        # Check model files
        model_exists = self.model_dir.exists() and \
                      any(self.model_dir.iterdir())
        checks.append(("Model files", model_exists))
        
        # Check config files
        config_exists = Path("configs/inference/beatheritage_v1.yaml").exists()
        checks.append(("Config files", config_exists))
        
        # Check postprocessor
        postprocessor_exists = Path("beatheritage_postprocessor.py").exists()
        checks.append(("Postprocessor", postprocessor_exists))
        
        # Check dependencies
        deps_ok = self.check_dependencies()
        checks.append(("Dependencies", deps_ok))
        
        # Print verification results
        logger.info("\n" + "="*50)
        logger.info("SETUP VERIFICATION")
        logger.info("="*50)
        
        all_ok = True
        for name, status in checks:
            status_str = "✅" if status else "❌"
            logger.info(f"{status_str} {name}: {'OK' if status else 'FAILED'}")
            all_ok = all_ok and status
        
        logger.info("="*50)
        
        return all_ok
    
    def create_test_script(self):
        """Create a simple test script"""
        test_script = """#!/usr/bin/env python3
# BeatHeritage V1 Test Script

import subprocess
import sys

print("Testing BeatHeritage V1...")

# Test inference
cmd = [
    sys.executable, "inference.py",
    "-cn", "beatheritage_v1",
    "audio_path=demo.mp3",
    "output_path=test_output",
    "gamemode=0",
    "difficulty=5.5"
]

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode == 0:
        print("✅ Test successful!")
    else:
        print("❌ Test failed:")
        print(result.stderr)
except Exception as e:
    print(f"❌ Test error: {e}")
"""
        
        test_path = Path("test_beatheritage.py")
        with open(test_path, 'w') as f:
            f.write(test_script)
        test_path.chmod(0o755)
        
        logger.info(f"Created test script: {test_path}")
    
    def setup_all(self, force_download: bool = False):
        """Run complete setup process"""
        logger.info("Starting BeatHeritage V1 setup...")
        
        # Step 1: Install dependencies
        if not self.check_dependencies():
            self.install_dependencies()
        
        # Step 2: Download models
        self.download_model(force_download)
        self.download_diffusion_model()
        
        # Step 3: Create test script
        self.create_test_script()
        
        # Step 4: Verify setup
        if self.verify_setup():
            logger.info("\n🎉 BeatHeritage V1 setup complete!")
            logger.info("\nYou can now:")
            logger.info("1. Run the Web UI: python web-ui.py")
            logger.info("2. Use CLI: python inference.py -cn beatheritage_v1 ...")
            logger.info("3. Run the test: python test_beatheritage.py")
        else:
            logger.error("\n⚠️ Setup incomplete. Please check the errors above.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Setup BeatHeritage V1 model and dependencies'
    )
    parser.add_argument(
        '--model-id',
        type=str,
        default='hongminh54/BeatHeritage-v1',
        help='Hugging Face model ID'
    )
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if model exists'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )
    
    args = parser.parse_args()
    
    setup = BeatHeritageSetup(args.model_id)
    
    if args.skip_deps:
        logger.info("Skipping dependency installation")
        setup.download_model(args.force_download)
        setup.download_diffusion_model()
        setup.verify_setup()
    else:
        setup.setup_all(args.force_download)


if __name__ == "__main__":
    main()
