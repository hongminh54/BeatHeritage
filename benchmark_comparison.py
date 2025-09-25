#!/usr/bin/env python3
"""
BeatHeritage V1 vs Mapperatorinator V30 Benchmark Script
Compares performance, quality, and generation characteristics
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmarks comparing BeatHeritage V1 with Mapperatorinator V30"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        
    def run_inference(self, model_config: str, audio_path: str, 
                     gamemode: int, difficulty: float) -> Dict:
        """Run inference with specified model and parameters"""
        
        output_path = self.output_dir / f"{model_config}_{Path(audio_path).stem}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'python', 'inference.py',
            '-cn', model_config,
            f'audio_path={audio_path}',
            f'output_path={str(output_path)}',
            f'gamemode={gamemode}',
            f'difficulty={difficulty}',
        ]
        
        # Add model-specific parameters
        if model_config == 'beatheritage_v1':
            cmd.extend([
                'temperature=0.85',
                'top_p=0.92',
                'quality_control.enable_auto_correction=true',
                'quality_control.enable_flow_optimization=true',
                'advanced_features.enable_pattern_variety=true',
            ])
        else:  # v30
            cmd.extend([
                'temperature=0.9',
                'top_p=0.9',
            ])
        
        # Measure performance
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            # Parse output for quality metrics
            output_files = list(output_path.glob('*.osu'))
            
            metrics = {
                'model': model_config,
                'audio': Path(audio_path).name,
                'gamemode': gamemode,
                'difficulty': difficulty,
                'generation_time': end_time - start_time,
                'memory_usage': memory_after - memory_before,
                'success': True,
                'output_files': len(output_files),
                'quality_metrics': self._analyze_quality(output_files[0] if output_files else None)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {model_config}: {e}")
            metrics = {
                'model': model_config,
                'audio': Path(audio_path).name,
                'gamemode': gamemode,
                'difficulty': difficulty,
                'generation_time': -1,
                'memory_usage': -1,
                'success': False,
                'error': str(e),
                'output_files': 0,
                'quality_metrics': {}
            }
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def _analyze_quality(self, osu_file: Optional[Path]) -> Dict:
        """Analyze quality metrics of generated beatmap"""
        if not osu_file or not osu_file.exists():
            return {}
        
        metrics = {
            'object_count': 0,
            'avg_spacing': 0,
            'spacing_variance': 0,
            'pattern_diversity': 0,
            'flow_score': 0,
            'difficulty_consistency': 0
        }
        
        try:
            with open(osu_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse hit objects
            hit_objects = []
            in_hit_objects = False
            
            for line in lines:
                if '[HitObjects]' in line:
                    in_hit_objects = True
                    continue
                
                if in_hit_objects and line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        try:
                            x, y = int(parts[0]), int(parts[1])
                            hit_objects.append((x, y))
                        except:
                            pass
            
            metrics['object_count'] = len(hit_objects)
            
            if len(hit_objects) > 1:
                # Calculate spacing metrics
                distances = []
                for i in range(1, len(hit_objects)):
                    dist = np.sqrt(
                        (hit_objects[i][0] - hit_objects[i-1][0])**2 +
                        (hit_objects[i][1] - hit_objects[i-1][1])**2
                    )
                    distances.append(dist)
                
                metrics['avg_spacing'] = np.mean(distances)
                metrics['spacing_variance'] = np.var(distances)
                
                # Pattern diversity (entropy of distance distribution)
                hist, _ = np.histogram(distances, bins=10)
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                metrics['pattern_diversity'] = entropy
                
                # Flow score (based on angle changes)
                if len(hit_objects) > 2:
                    angles = []
                    for i in range(2, len(hit_objects)):
                        angle = self._calculate_angle(
                            hit_objects[i-2], 
                            hit_objects[i-1], 
                            hit_objects[i]
                        )
                        angles.append(angle)
                    
                    # Lower angle variance = better flow
                    metrics['flow_score'] = 1.0 / (1.0 + np.var(angles) / 100)
                
                # Difficulty consistency
                chunk_size = max(10, len(distances) // 10)
                chunk_variances = []
                for i in range(0, len(distances), chunk_size):
                    chunk = distances[i:i+chunk_size]
                    if chunk:
                        chunk_variances.append(np.var(chunk))
                
                if chunk_variances:
                    metrics['difficulty_consistency'] = 1.0 / (1.0 + np.var(chunk_variances))
        
        except Exception as e:
            logger.error(f"Error analyzing quality: {e}")
        
        return metrics
    
    def _calculate_angle(self, p1: Tuple, p2: Tuple, p3: Tuple) -> float:
        """Calculate angle between three points"""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        
        angle_diff = angle2 - angle1
        # Normalize to [-pi, pi]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        return abs(angle_diff)
    
    def run_benchmark_suite(self, test_audio_files: List[str]):
        """Run complete benchmark suite"""
        
        models = ['beatheritage_v1', 'v30']
        gamemodes = [0, 1, 2, 3]  # All gamemodes
        difficulties = [3.0, 5.5, 7.5]  # Easy, Normal, Hard
        
        total_tests = len(test_audio_files) * len(models) * len(gamemodes) * len(difficulties)
        
        with tqdm(total=total_tests, desc="Running benchmarks") as pbar:
            for audio_file in test_audio_files:
                for gamemode in gamemodes:
                    for difficulty in difficulties:
                        for model in models:
                            logger.info(f"Testing {model} on {audio_file} "
                                      f"(GM:{gamemode}, Diff:{difficulty})")
                            
                            result = self.run_inference(
                                model, audio_file, gamemode, difficulty
                            )
                            self.results.append(result)
                            pbar.update(1)
                            
                            # Save intermediate results
                            self._save_results()
    
    def _save_results(self):
        """Save benchmark results to JSON and CSV"""
        # Save as JSON
        json_path = self.output_dir / f"benchmark_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV for analysis
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / f"benchmark_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def generate_report(self):
        """Generate comprehensive benchmark report with visualizations"""
        
        if not self.results:
            logger.error("No results to generate report")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create visualizations
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Generation Time Comparison
        ax1 = plt.subplot(2, 3, 1)
        successful_df = df[df['success'] == True]
        if not successful_df.empty:
            sns.boxplot(data=successful_df, x='model', y='generation_time', ax=ax1)
            ax1.set_title('Generation Time Comparison')
            ax1.set_ylabel('Time (seconds)')
        
        # 2. Memory Usage Comparison
        ax2 = plt.subplot(2, 3, 2)
        if not successful_df.empty:
            sns.boxplot(data=successful_df, x='model', y='memory_usage', ax=ax2)
            ax2.set_title('Memory Usage Comparison')
            ax2.set_ylabel('Memory (MB)')
        
        # 3. Success Rate
        ax3 = plt.subplot(2, 3, 3)
        success_rates = df.groupby('model')['success'].mean() * 100
        success_rates.plot(kind='bar', ax=ax3)
        ax3.set_title('Success Rate (%)')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim(0, 105)
        
        # 4. Quality Metrics Comparison
        if not successful_df.empty and 'quality_metrics' in successful_df.columns:
            # Extract quality metrics
            quality_data = []
            for _, row in successful_df.iterrows():
                if row['quality_metrics']:
                    quality_data.append({
                        'model': row['model'],
                        'pattern_diversity': row['quality_metrics'].get('pattern_diversity', 0),
                        'flow_score': row['quality_metrics'].get('flow_score', 0),
                        'difficulty_consistency': row['quality_metrics'].get('difficulty_consistency', 0)
                    })
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                
                # Pattern Diversity
                ax4 = plt.subplot(2, 3, 4)
                if 'pattern_diversity' in quality_df.columns:
                    sns.boxplot(data=quality_df, x='model', y='pattern_diversity', ax=ax4)
                    ax4.set_title('Pattern Diversity Score')
                
                # Flow Score
                ax5 = plt.subplot(2, 3, 5)
                if 'flow_score' in quality_df.columns:
                    sns.boxplot(data=quality_df, x='model', y='flow_score', ax=ax5)
                    ax5.set_title('Flow Quality Score')
                
                # Difficulty Consistency
                ax6 = plt.subplot(2, 3, 6)
                if 'difficulty_consistency' in quality_df.columns:
                    sns.boxplot(data=quality_df, x='model', y='difficulty_consistency', ax=ax6)
                    ax6.set_title('Difficulty Consistency Score')
        
        plt.suptitle('BeatHeritage V1 vs Mapperatorinator V30 Benchmark Report', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"benchmark_report_{self.timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Generate text summary
        summary = self._generate_text_summary(df)
        summary_path = self.output_dir / f"benchmark_summary_{self.timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Report generated: {plot_path} and {summary_path}")
    
    def _generate_text_summary(self, df: pd.DataFrame) -> str:
        """Generate text summary of benchmark results"""
        
        summary = []
        summary.append("=" * 80)
        summary.append("BEATHERITAGE V1 VS MAPPERATORINATOR V30 BENCHMARK SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Timestamp: {self.timestamp}")
        summary.append(f"Total Tests: {len(df)}")
        summary.append("")
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            successful_df = model_df[model_df['success'] == True]
            
            summary.append(f"\n{model.upper()}")
            summary.append("-" * 40)
            summary.append(f"Success Rate: {model_df['success'].mean()*100:.1f}%")
            
            if not successful_df.empty:
                summary.append(f"Avg Generation Time: {successful_df['generation_time'].mean():.2f}s")
                summary.append(f"Avg Memory Usage: {successful_df['memory_usage'].mean():.1f}MB")
                
                # Quality metrics
                quality_metrics = []
                for _, row in successful_df.iterrows():
                    if row['quality_metrics']:
                        quality_metrics.append(row['quality_metrics'])
                
                if quality_metrics:
                    avg_diversity = np.mean([m.get('pattern_diversity', 0) for m in quality_metrics])
                    avg_flow = np.mean([m.get('flow_score', 0) for m in quality_metrics])
                    avg_consistency = np.mean([m.get('difficulty_consistency', 0) for m in quality_metrics])
                    
                    summary.append(f"Avg Pattern Diversity: {avg_diversity:.3f}")
                    summary.append(f"Avg Flow Score: {avg_flow:.3f}")
                    summary.append(f"Avg Difficulty Consistency: {avg_consistency:.3f}")
        
        # Winner determination
        summary.append("\n" + "=" * 80)
        summary.append("WINNER ANALYSIS")
        summary.append("=" * 80)
        
        if len(df['model'].unique()) == 2:
            model1, model2 = df['model'].unique()
            
            # Compare metrics
            metrics_comparison = []
            
            for metric in ['generation_time', 'memory_usage']:
                m1_avg = df[df['model'] == model1][metric].mean()
                m2_avg = df[df['model'] == model2][metric].mean()
                
                if m1_avg < m2_avg:
                    winner = model1
                    improvement = ((m2_avg - m1_avg) / m2_avg) * 100
                else:
                    winner = model2
                    improvement = ((m1_avg - m2_avg) / m1_avg) * 100
                
                metrics_comparison.append(
                    f"{metric}: {winner} ({improvement:.1f}% better)"
                )
            
            for comp in metrics_comparison:
                summary.append(comp)
        
        return "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(description='Benchmark BeatHeritage V1 vs V30')
    parser.add_argument(
        '--audio-dir', 
        type=str, 
        default='./test_audio',
        help='Directory containing test audio files'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./benchmark_results',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited parameters'
    )
    
    args = parser.parse_args()
    
    # Get test audio files
    audio_dir = Path(args.audio_dir)
    if audio_dir.exists():
        audio_files = list(audio_dir.glob('*.mp3')) + list(audio_dir.glob('*.ogg'))
    else:
        # Use demo files
        logger.warning(f"Audio directory {audio_dir} not found, using demo files")
        audio_files = ['demo.mp3']  # Fallback to demo
    
    if args.quick_test:
        # Quick test with limited parameters
        audio_files = audio_files[:1]
        logger.info("Running quick test with 1 audio file")
    
    # Run benchmarks
    runner = BenchmarkRunner(args.output_dir)
    runner.run_benchmark_suite([str(f) for f in audio_files])
    runner.generate_report()
    
    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
