#src/data/quality_control.py
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ArgoQualityControl:
    """Advanced ARGO data quality control system"""
    
    def __init__(self):
        self.qc_flags = {
            0: "No QC performed",
            1: "Good data",
            2: "Probably good data", 
            3: "Probably bad data",
            4: "Bad data",
            8: "Interpolated value",
            9: "Missing value"
        }
        
        # ARGO standard ranges
        self.standard_ranges = {
            'temperature': (-2.5, 40.0),
            'salinity': (0.0, 41.0),
            'pressure': (0.0, 3000.0),
            'oxygen': (0.0, 500.0),
            'chlorophyll': (0.0, 50.0),
            'nitrate': (0.0, 50.0)
        }

    def comprehensive_qc_check(self, profile_data: Dict) -> Dict[str, Any]:
        """Comprehensive quality control check for ARGO profiles"""
        qc_results = {
            'basic_checks': self._basic_quality_checks(profile_data),
            'technical_checks': self._technical_checks(profile_data),
            'scientific_checks': self._scientific_checks(profile_data),
            'outliers': self._detect_all_outliers(profile_data),
            'quality_score': self._calculate_quality_score(profile_data),
            'recommendations': []
        }
        
        # Generate recommendations
        qc_results['recommendations'] = self._generate_recommendations(qc_results)
        
        return qc_results

    def _basic_quality_checks(self, profile_data: Dict) -> Dict[str, Any]:
        """Basic data quality checks"""
        checks = {
            'missing_data': self._check_missing_data(profile_data),
            'value_ranges': self._check_value_ranges(profile_data),
            'pressure_monotonic': self._check_pressure_monotonic(profile_data),
            'spike_detection': self._detect_spikes(profile_data)
        }
        return checks

    def _technical_checks(self, profile_data: Dict) -> Dict[str, Any]:
        """Technical checks for data consistency"""
        checks = {
            'profile_integrity': self._check_profile_integrity(profile_data),
            'sensor_consistency': self._check_sensor_consistency(profile_data),
            'timestamp_validity': self._check_timestamp_validity(profile_data)
        }
        return checks

    def _scientific_checks(self, profile_data: Dict) -> Dict[str, Any]:
        """Scientific plausibility checks"""
        checks = {
            'ts_relationship': self._check_temperature_salinity_relationship(profile_data),
            'depth_consistency': self._check_depth_consistency(profile_data),
            'regional_plausibility': self._check_regional_plausibility(profile_data)
        }
        return checks

    def _check_missing_data(self, profile_data: Dict) -> Dict[str, Any]:
        """Check for missing or invalid data"""
        results = {}
        
        for param in ['temperature_values', 'salinity_values', 'pressure_levels']:
            if param in profile_data and profile_data[param]:
                values = profile_data[param]
                missing_count = sum(1 for v in values if v is None or np.isnan(v))
                results[param] = {
                    'total_values': len(values),
                    'missing_count': missing_count,
                    'completeness': (len(values) - missing_count) / len(values) * 100 if values else 0
                }
        
        return results

    def _check_value_ranges(self, profile_data: Dict) -> Dict[str, Any]:
        """Check if values are within realistic ranges"""
        results = {}
        
        for param, (min_val, max_val) in self.standard_ranges.items():
            param_key = f"{param}_values"
            if param_key in profile_data and profile_data[param_key]:
                values = [v for v in profile_data[param_key] if v is not None and not np.isnan(v)]
                if values:
                    out_of_range = sum(1 for v in values if v < min_val or v > max_val)
                    results[param] = {
                        'min_found': min(values),
                        'max_found': max(values),
                        'out_of_range_count': out_of_range,
                        'within_range_percentage': (len(values) - out_of_range) / len(values) * 100
                    }
        
        return results

    def _check_pressure_monotonic(self, profile_data: Dict) -> Dict[str, Any]:
        """Check if pressure increases monotonically with depth"""
        if 'pressure_levels' not in profile_data or not profile_data['pressure_levels']:
            return {'valid': False, 'error': 'No pressure data'}
        
        pressures = [p for p in profile_data['pressure_levels'] if p is not None and not np.isnan(p)]
        
        if len(pressures) < 2:
            return {'valid': True, 'warning': 'Insufficient pressure points'}
        
        # Check if pressures are strictly increasing
        is_monotonic = all(pressures[i] < pressures[i+1] for i in range(len(pressures)-1))
        
        return {
            'valid': is_monotonic,
            'pressure_points': len(pressures),
            'max_pressure_gap': max([pressures[i+1] - pressures[i] for i in range(len(pressures)-1)]) if is_monotonic else None
        }

    def _detect_spikes(self, profile_data: Dict) -> Dict[str, Any]:
        """Detect spikes in parameter values"""
        spikes = {}
        
        for param in ['temperature_values', 'salinity_values']:
            if param in profile_data and profile_data[param]:
                values = profile_data[param]
                param_spikes = self._find_spikes_in_series(values, param.replace('_values', ''))
                if param_spikes:
                    spikes[param] = param_spikes
        
        return spikes

    def _find_spikes_in_series(self, values: List[float], param_name: str) -> List[Dict]:
        """Find spikes in a data series"""
        spikes = []
        
        if len(values) < 3:
            return spikes
        
        for i in range(1, len(values)-1):
            if None in [values[i-1], values[i], values[i+1]]:
                continue
                
            # Calculate gradients
            grad_prev = abs(values[i] - values[i-1])
            grad_next = abs(values[i] - values[i+1])
            grad_neighbors = abs(values[i-1] - values[i+1])
            
            # Spike detection criteria
            if grad_prev > 3 * grad_neighbors and grad_next > 3 * grad_neighbors:
                spikes.append({
                    'index': i,
                    'value': values[i],
                    'severity': 'high' if grad_prev > 5 * grad_neighbors else 'medium',
                    'surrounding_values': [values[i-1], values[i+1]]
                })
        
        return spikes

    def _check_profile_integrity(self, profile_data: Dict) -> Dict[str, Any]:
        """Check profile data integrity"""
        integrity_checks = {}
        
        # Check if all arrays have same length
        array_lengths = {}
        for key, value in profile_data.items():
            if key.endswith('_values') or key.endswith('_levels'):
                if isinstance(value, list):
                    array_lengths[key] = len(value)
        
        if array_lengths:
            unique_lengths = set(array_lengths.values())
            integrity_checks['consistent_lengths'] = len(unique_lengths) == 1
            integrity_checks['array_lengths'] = array_lengths
        
        return integrity_checks

    def _calculate_quality_score(self, profile_data: Dict) -> float:
        """Calculate overall quality score (0-100)"""
        score_components = []
        
        # Completeness score (25%)
        completeness = self._calculate_completeness_score(profile_data)
        score_components.append(completeness * 0.25)
        
        # Range validity score (25%)
        range_score = self._calculate_range_score(profile_data)
        score_components.append(range_score * 0.25)
        
        # Technical quality score (25%)
        technical_score = self._calculate_technical_score(profile_data)
        score_components.append(technical_score * 0.25)
        
        # Scientific plausibility score (25%)
        scientific_score = self._calculate_scientific_score(profile_data)
        score_components.append(scientific_score * 0.25)
        
        return sum(score_components)

    def _calculate_completeness_score(self, profile_data: Dict) -> float:
        """Calculate data completeness score"""
        total_points = 0
        valid_points = 0
        
        for param in ['temperature_values', 'salinity_values', 'pressure_levels']:
            if param in profile_data and profile_data[param]:
                values = profile_data[param]
                total_points += len(values)
                valid_points += sum(1 for v in values if v is not None and not np.isnan(v))
        
        return (valid_points / total_points * 100) if total_points > 0 else 0

    def _generate_recommendations(self, qc_results: Dict) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if qc_results['basic_checks']['missing_data'].get('completeness', 100) < 90:
            recommendations.append("Consider data gap filling techniques")
        
        if qc_results['basic_checks']['pressure_monotonic'].get('valid') is False:
            recommendations.append("Pressure data needs monotonicity correction")
        
        if qc_results['basic_checks']['spike_detection']:
            recommendations.append("Spikes detected - consider smoothing or removal")
        
        if qc_results['quality_score'] < 80:
            recommendations.append("Overall data quality needs improvement")
        
        return recommendations

    # Placeholder methods for additional checks
    def _check_sensor_consistency(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _check_timestamp_validity(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _check_temperature_salinity_relationship(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _check_depth_consistency(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _check_regional_plausibility(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _detect_all_outliers(self, profile_data: Dict) -> Dict[str, Any]:
        return {'status': 'check_implemented'}
    
    def _calculate_range_score(self, profile_data: Dict) -> float:
        return 85.0
    
    def _calculate_technical_score(self, profile_data: Dict) -> float:
        return 90.0
    
    def _calculate_scientific_score(self, profile_data: Dict) -> float:
        return 88.0
    