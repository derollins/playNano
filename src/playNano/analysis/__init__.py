"""Public package initialization. Analysis modules live here."""

# Import built-in modules so they register or can be referenced
from .modules.count_nonzero import CountNonzeroModule
from .modules.feature_detection import FeatureDetectionModule

# import other built-in modules implemented, e.g.:
# from .modules.detection import ParticleDetector
# from .modules.segmentation import FrameSegmenter
# ...

# Build registry: map module.name to class
_BUILTIN = [
    CountNonzeroModule,
    FeatureDetectionModule,
    # FrameSegmenter,
    # etc.
]

BUILTIN_ANALYSIS_MODULES = {cls().name: cls for cls in _BUILTIN}
