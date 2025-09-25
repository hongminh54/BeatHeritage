"""
BeatHeritage V1 Custom Postprocessor
Enhanced postprocessing for improved beatmap quality
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

from osuT5.osuT5.inference.postprocessor import Postprocessor, BeatmapConfig

logger = logging.getLogger(__name__)


@dataclass
class BeatHeritageConfig(BeatmapConfig):
    """Enhanced config for BeatHeritage V1 postprocessing"""
    # Quality control parameters
    min_distance_threshold: float = 20.0
    max_overlap_ratio: float = 0.15
    enable_auto_correction: bool = True
    enable_flow_optimization: bool = True
    
    # Pattern enhancement
    enable_pattern_variety: bool = True
    pattern_complexity_target: float = 0.7
    
    # Difficulty scaling
    enable_difficulty_scaling: bool = True
    difficulty_variance_threshold: float = 0.3
    
    # Style preservation
    enable_style_preservation: bool = True
    style_consistency_weight: float = 0.8


class BeatHeritagePostprocessor(Postprocessor):
    """Enhanced postprocessor for BeatHeritage V1"""
    
    def __init__(self, config: BeatHeritageConfig):
        super().__init__(config)
        self.config = config
        self.flow_optimizer = FlowOptimizer(config)
        self.pattern_enhancer = PatternEnhancer(config)
        self.quality_controller = QualityController(config)
    
    def postprocess(self, beatmap_data: Dict) -> Dict:
        """
        Enhanced postprocessing pipeline for BeatHeritage V1
        
        Args:
            beatmap_data: Raw beatmap data from model
            
        Returns:
            Processed beatmap data with enhancements
        """
        # Base postprocessing
        beatmap_data = super().postprocess(beatmap_data)
        
        # Quality control
        if self.config.enable_auto_correction:
            beatmap_data = self.quality_controller.fix_spacing_issues(beatmap_data)
            beatmap_data = self.quality_controller.fix_overlaps(beatmap_data)
        
        # Flow optimization
        if self.config.enable_flow_optimization:
            beatmap_data = self.flow_optimizer.optimize_flow(beatmap_data)
        
        # Pattern enhancement
        if self.config.enable_pattern_variety:
            beatmap_data = self.pattern_enhancer.enhance_patterns(beatmap_data)
        
        # Difficulty scaling
        if self.config.enable_difficulty_scaling:
            beatmap_data = self._scale_difficulty(beatmap_data)
        
        # Style preservation
        if self.config.enable_style_preservation:
            beatmap_data = self._preserve_style(beatmap_data)
        
        return beatmap_data
    
    def _scale_difficulty(self, beatmap_data: Dict) -> Dict:
        """Scale difficulty to match target star rating"""
        target_difficulty = self.config.difficulty
        if target_difficulty is None:
            return beatmap_data
        
        current_difficulty = self._calculate_difficulty(beatmap_data)
        scale_factor = target_difficulty / max(current_difficulty, 0.1)
        
        # Adjust spacing and timing based on scale factor
        if 'hit_objects' in beatmap_data:
            for obj in beatmap_data['hit_objects']:
                if 'distance' in obj:
                    obj['distance'] *= scale_factor
        
        logger.info(f"Scaled difficulty from {current_difficulty:.2f} to {target_difficulty:.2f}")
        return beatmap_data
    
    def _preserve_style(self, beatmap_data: Dict) -> Dict:
        """Preserve mapping style consistency"""
        # Analyze style characteristics
        style_features = self._extract_style_features(beatmap_data)
        
        # Apply style consistency
        consistency_weight = self.config.style_consistency_weight
        
        if 'hit_objects' in beatmap_data:
            for i, obj in enumerate(beatmap_data['hit_objects']):
                if i > 0:
                    # Maintain consistent spacing patterns
                    prev_obj = beatmap_data['hit_objects'][i-1]
                    expected_distance = style_features.get('avg_distance', 100)
                    
                    if 'position' in obj and 'position' in prev_obj:
                        current_distance = self._calculate_distance(
                            obj['position'], prev_obj['position']
                        )
                        
                        # Blend current with expected based on consistency weight
                        adjusted_distance = (
                            current_distance * (1 - consistency_weight) +
                            expected_distance * consistency_weight
                        )
                        
                        # Adjust position to match distance
                        obj['position'] = self._adjust_position(
                            prev_obj['position'],
                            obj['position'],
                            adjusted_distance
                        )
        
        return beatmap_data
    
    def _calculate_difficulty(self, beatmap_data: Dict) -> float:
        """Calculate approximate star rating"""
        # Simplified difficulty calculation
        num_objects = len(beatmap_data.get('hit_objects', []))
        avg_spacing = self._calculate_avg_spacing(beatmap_data)
        bpm = beatmap_data.get('bpm', 180)
        
        # Simple formula (can be improved)
        difficulty = (num_objects / 100) * (avg_spacing / 50) * (bpm / 180)
        return min(max(difficulty, 0), 10)  # Clamp to 0-10
    
    def _extract_style_features(self, beatmap_data: Dict) -> Dict:
        """Extract style characteristics from beatmap"""
        features = {}
        
        if 'hit_objects' in beatmap_data:
            distances = []
            for i in range(1, len(beatmap_data['hit_objects'])):
                if 'position' in beatmap_data['hit_objects'][i]:
                    dist = self._calculate_distance(
                        beatmap_data['hit_objects'][i-1].get('position', (256, 192)),
                        beatmap_data['hit_objects'][i]['position']
                    )
                    distances.append(dist)
            
            if distances:
                features['avg_distance'] = np.mean(distances)
                features['distance_variance'] = np.var(distances)
        
        return features
    
    def _calculate_avg_spacing(self, beatmap_data: Dict) -> float:
        """Calculate average spacing between objects"""
        distances = []
        objects = beatmap_data.get('hit_objects', [])
        
        for i in range(1, len(objects)):
            if 'position' in objects[i] and 'position' in objects[i-1]:
                dist = self._calculate_distance(
                    objects[i-1]['position'],
                    objects[i]['position']
                )
                distances.append(dist)
        
        return np.mean(distances) if distances else 100
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _adjust_position(self, from_pos: Tuple[float, float],
                        to_pos: Tuple[float, float],
                        target_distance: float) -> Tuple[float, float]:
        """Adjust position to achieve target distance"""
        current_distance = self._calculate_distance(from_pos, to_pos)
        if current_distance < 0.01:  # Avoid division by zero
            return to_pos
        
        scale = target_distance / current_distance
        dx = (to_pos[0] - from_pos[0]) * scale
        dy = (to_pos[1] - from_pos[1]) * scale
        
        # Keep within playfield bounds
        new_x = max(0, min(512, from_pos[0] + dx))
        new_y = max(0, min(384, from_pos[1] + dy))
        
        return (new_x, new_y)


class FlowOptimizer:
    """Optimize flow patterns in beatmaps"""
    
    def __init__(self, config: BeatHeritageConfig):
        self.config = config
    
    def optimize_flow(self, beatmap_data: Dict) -> Dict:
        """Optimize flow for better playability"""
        if 'hit_objects' not in beatmap_data:
            return beatmap_data
        
        objects = beatmap_data['hit_objects']
        optimized_objects = []
        
        for i, obj in enumerate(objects):
            if i >= 2 and 'position' in obj:
                # Calculate flow angle
                prev_angle = self._calculate_angle(
                    objects[i-2].get('position', (256, 192)),
                    objects[i-1].get('position', (256, 192))
                )
                current_angle = self._calculate_angle(
                    objects[i-1].get('position', (256, 192)),
                    obj['position']
                )
                
                # Smooth sharp angles
                angle_diff = abs(current_angle - prev_angle)
                if angle_diff > 120:  # Sharp angle threshold
                    # Adjust position for smoother flow
                    smoothed_angle = prev_angle + np.sign(current_angle - prev_angle) * 90
                    distance = self._calculate_distance(
                        objects[i-1]['position'],
                        obj['position']
                    )
                    
                    new_x = objects[i-1]['position'][0] + distance * np.cos(np.radians(smoothed_angle))
                    new_y = objects[i-1]['position'][1] + distance * np.sin(np.radians(smoothed_angle))
                    
                    obj['position'] = (
                        max(0, min(512, new_x)),
                        max(0, min(384, new_y))
                    )
            
            optimized_objects.append(obj)
        
        beatmap_data['hit_objects'] = optimized_objects
        return beatmap_data
    
    def _calculate_angle(self, pos1: Tuple[float, float],
                        pos2: Tuple[float, float]) -> float:
        """Calculate angle between two positions in degrees"""
        return np.degrees(np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]))
    
    def _calculate_distance(self, pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


class PatternEnhancer:
    """Enhance pattern variety in beatmaps"""
    
    def __init__(self, config: BeatHeritageConfig):
        self.config = config
        self.pattern_library = self._load_pattern_library()
    
    def enhance_patterns(self, beatmap_data: Dict) -> Dict:
        """Enhance patterns for more variety"""
        if 'hit_objects' not in beatmap_data:
            return beatmap_data
        
        # Detect repetitive patterns
        repetitive_sections = self._detect_repetitive_patterns(beatmap_data)
        
        # Replace with varied patterns
        for section in repetitive_sections:
            beatmap_data = self._vary_pattern(beatmap_data, section)
        
        return beatmap_data
    
    def _load_pattern_library(self) -> List[Dict]:
        """Load common mapping patterns"""
        return [
            {'name': 'triangle', 'positions': [(0, 0), (100, 0), (50, 86.6)]},
            {'name': 'square', 'positions': [(0, 0), (100, 0), (100, 100), (0, 100)]},
            {'name': 'star', 'positions': [(50, 0), (61, 35), (97, 35), (68, 57), (79, 91), (50, 70), (21, 91), (32, 57), (3, 35), (39, 35)]},
            {'name': 'hexagon', 'positions': [(50, 0), (93, 25), (93, 75), (50, 100), (7, 75), (7, 25)]},
        ]
    
    def _detect_repetitive_patterns(self, beatmap_data: Dict) -> List[Tuple[int, int]]:
        """Detect sections with repetitive patterns"""
        repetitive_sections = []
        objects = beatmap_data.get('hit_objects', [])
        
        window_size = 8
        for i in range(len(objects) - window_size * 2):
            pattern1 = self._extract_pattern(objects[i:i+window_size])
            pattern2 = self._extract_pattern(objects[i+window_size:i+window_size*2])
            
            if self._patterns_similar(pattern1, pattern2):
                repetitive_sections.append((i, i + window_size * 2))
        
        return repetitive_sections
    
    def _extract_pattern(self, objects: List[Dict]) -> List[Tuple[float, float]]:
        """Extract position pattern from objects"""
        return [obj.get('position', (256, 192)) for obj in objects]
    
    def _patterns_similar(self, pattern1: List, pattern2: List, threshold: float = 0.8) -> bool:
        """Check if two patterns are similar"""
        if len(pattern1) != len(pattern2):
            return False
        
        distances = []
        for pos1, pos2 in zip(pattern1, pattern2):
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        return avg_distance < 50  # Threshold for similarity
    
    def _vary_pattern(self, beatmap_data: Dict, section: Tuple[int, int]) -> Dict:
        """Apply variation to a pattern section"""
        start, end = section
        objects = beatmap_data['hit_objects']
        
        # Select a random pattern from library
        pattern = np.random.choice(self.pattern_library)
        pattern_positions = pattern['positions']
        
        # Apply pattern with scaling
        section_length = end - start
        for i in range(start, min(end, len(objects))):
            if 'position' in objects[i]:
                pattern_idx = (i - start) % len(pattern_positions)
                base_pos = pattern_positions[pattern_idx]
                
                # Scale and translate pattern
                center = (256, 192)
                scale = 2.0
                
                new_x = center[0] + base_pos[0] * scale
                new_y = center[1] + base_pos[1] * scale
                
                objects[i]['position'] = (
                    max(0, min(512, new_x)),
                    max(0, min(384, new_y))
                )
        
        return beatmap_data


class QualityController:
    """Control quality aspects of beatmaps"""
    
    def __init__(self, config: BeatHeritageConfig):
        self.config = config
    
    def fix_spacing_issues(self, beatmap_data: Dict) -> Dict:
        """Fix objects that are too close together"""
        if 'hit_objects' not in beatmap_data:
            return beatmap_data
        
        objects = beatmap_data['hit_objects']
        min_distance = self.config.min_distance_threshold
        
        for i in range(1, len(objects)):
            if 'position' in objects[i] and 'position' in objects[i-1]:
                distance = self._calculate_distance(
                    objects[i-1]['position'],
                    objects[i]['position']
                )
                
                if distance < min_distance:
                    # Move object to maintain minimum distance
                    direction = self._get_direction(
                        objects[i-1]['position'],
                        objects[i]['position']
                    )
                    
                    objects[i]['position'] = self._move_position(
                        objects[i-1]['position'],
                        direction,
                        min_distance
                    )
        
        return beatmap_data
    
    def fix_overlaps(self, beatmap_data: Dict) -> Dict:
        """Fix overlapping sliders and circles"""
        if 'hit_objects' not in beatmap_data:
            return beatmap_data
        
        objects = beatmap_data['hit_objects']
        max_overlap = self.config.max_overlap_ratio
        
        for i in range(len(objects)):
            for j in range(i+1, min(i+10, len(objects))):  # Check next 10 objects
                if self._objects_overlap(objects[i], objects[j], max_overlap):
                    # Adjust position to reduce overlap
                    objects[j] = self._adjust_for_overlap(objects[i], objects[j])
        
        return beatmap_data
    
    def _calculate_distance(self, pos1: Tuple[float, float],
                          pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_direction(self, from_pos: Tuple[float, float],
                      to_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Get normalized direction vector"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        length = np.sqrt(dx**2 + dy**2)
        if length < 0.01:
            return (1, 0)  # Default right direction
        
        return (dx / length, dy / length)
    
    def _move_position(self, from_pos: Tuple[float, float],
                      direction: Tuple[float, float],
                      distance: float) -> Tuple[float, float]:
        """Move position in direction by distance"""
        new_x = from_pos[0] + direction[0] * distance
        new_y = from_pos[1] + direction[1] * distance
        
        # Keep within bounds
        return (
            max(0, min(512, new_x)),
            max(0, min(384, new_y))
        )
    
    def _objects_overlap(self, obj1: Dict, obj2: Dict, threshold: float) -> bool:
        """Check if two objects overlap beyond threshold"""
        if 'position' not in obj1 or 'position' not in obj2:
            return False
        
        distance = self._calculate_distance(obj1['position'], obj2['position'])
        
        # Simple overlap check (can be improved for sliders)
        radius = 30  # Approximate circle radius
        overlap = max(0, 2 * radius - distance) / (2 * radius)
        
        return overlap > threshold
    
    def _adjust_for_overlap(self, obj1: Dict, obj2: Dict) -> Dict:
        """Adjust object position to reduce overlap"""
        if 'position' not in obj1 or 'position' not in obj2:
            return obj2
        
        # Move obj2 away from obj1
        direction = self._get_direction(obj1['position'], obj2['position'])
        min_safe_distance = 60  # Minimum safe distance
        
        obj2['position'] = self._move_position(
            obj1['position'],
            direction,
            min_safe_distance
        )
        
        return obj2


# Export main postprocessor
__all__ = ['BeatHeritagePostprocessor', 'BeatHeritageConfig']
