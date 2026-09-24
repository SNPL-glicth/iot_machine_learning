import os
import re
import shutil

mapping = {
    'anomaly': 'results.anomaly',
    'canonical_series': 'series.canonical_series',
    'change_point': 'patterns.change_point',
    'delta_spike': 'patterns.delta_spike',
    'memory_search_result': 'results.memory_search_result',
    'operational_regime': 'patterns.operational_regime',
    'pattern_result': 'patterns.pattern_result',
    'prediction': 'results.prediction',
    'sensor_ranges': 'iot.sensor_ranges',
    'sensor_reading': 'iot.sensor_reading',
    'series_context': 'series.series_context',
    'series_profile': 'series.series_profile',
    'structural_analysis': 'series.structural_analysis',
    'temporal_features': 'series.temporal_features',
    'time_series': 'series.time_series',
    'severity': 'results.severity',
    'threshold': 'series.threshold',
}

# Special mapping for pattern.py
pattern_classes = {
    'PatternType': 'patterns.pattern_result',
    'PatternResult': 'patterns.pattern_result',
    'ChangePointType': 'patterns.change_point',
    'ChangePoint': 'patterns.change_point',
    'SpikeClassification': 'patterns.delta_spike',
    'DeltaSpikeResult': 'patterns.delta_spike',
    'OperationalRegime': 'patterns.operational_regime',
}

# 1. Move implementation files
if os.path.exists('domain/entities/canonical_series.py'):
    shutil.move('domain/entities/canonical_series.py', 'domain/entities/series/canonical_series.py')
if os.path.exists('domain/entities/severity.py'):
    shutil.move('domain/entities/severity.py', 'domain/entities/results/severity.py')
if os.path.exists('domain/entities/threshold.py'):
    shutil.move('domain/entities/threshold.py', 'domain/entities/series/threshold.py')

# 2. Update all python files in the repo
for root, _, files in os.walk('.'):
    if '.venv' in root or '.git' in root: continue
    for file in files:
        if not file.endswith('.py'): continue
        filepath = os.path.join(root, file)
        with open(filepath, 'r') as f:
            content = f.read()
        
        orig_content = content
        
        # Replace simple module imports
        for old_mod, new_mod in mapping.items():
            # from iot_machine_learning.domain.entities.old_mod import ...
            content = re.sub(
                r'from iot_machine_learning\.domain\.entities\.' + old_mod + r' import',
                f'from iot_machine_learning.domain.entities.{new_mod} import',
                content
            )
            # from ...entities.old_mod import ...
            # Wait, relative imports are trickier, let's just do exact string matches where possible
            content = content.replace(f'from ..entities.{old_mod} import', f'from ..entities.{new_mod} import')
            content = content.replace(f'from .entities.{old_mod} import', f'from .entities.{new_mod} import')
        
        # Handle pattern.py
        for cls, new_mod in pattern_classes.items():
            content = re.sub(
                r'from iot_machine_learning\.domain\.entities\.pattern import(.*)' + cls,
                r'from iot_machine_learning.domain.entities.' + new_mod + r' import\1' + cls,
                content
            )
        
        # Also need to fix domain/entities/__init__.py
        if filepath.endswith('domain/entities/__init__.py'):
            content = content.replace('from .severity import SeverityResult', 'from .results.severity import SeverityResult')
            content = content.replace('from .threshold import Threshold', 'from .series.threshold import Threshold')
            content = content.replace('from .canonical_series', 'from .series.canonical_series')
            
        # Also need to fix domain/entities/series/__init__.py for canonical_series and threshold
        if filepath.endswith('domain/entities/series/__init__.py'):
            content = content.replace('from ..canonical_series', 'from .canonical_series')
            
        # Also need to fix domain/entities/series/series_context.py
        if filepath.endswith('domain/entities/series/series_context.py'):
            content = content.replace('from ..threshold import', 'from .threshold import')

        if content != orig_content:
            with open(filepath, 'w') as f:
                f.write(content)

# 3. Delete all facade files in domain/entities/
for f in os.listdir('domain/entities'):
    if f.endswith('.py') and f != '__init__.py':
        os.remove(os.path.join('domain/entities', f))

print("Refactoring complete.")
