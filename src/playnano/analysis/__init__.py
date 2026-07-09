"""Public package initialization. Analysis modules live here."""

# Import built-in modules so they register or can be referenced
from .modules.count_nonzero import CountNonzeroModule
from .modules.dbscan_clustering import DBSCANClusteringModule
from .modules.feature_detection import FeatureDetectionModule
from .modules.k_means_clustering import KMeansClusteringModule
from .modules.log_blob_detection import LoGBlobDetectionModule
from .modules.particle_extraction import ParticleRegionExtractionModule
from .modules.particle_region_media_export import ParticleRegionMediaExportModule
from .modules.particle_tracking import ParticleTrackingModule
from .modules.tracked_particle_boundary_size import BoundarySizeModule
from .modules.x_means_clustering import XMeansClusteringModule

# import other built-in modules implemented, e.g.:
# from .modules.detection import ParticleDetector
# from .modules.segmentation import FrameSegmenter
# ...

# Build registry: map module.name to class
_BUILTIN = [
    CountNonzeroModule,
    FeatureDetectionModule,
    LoGBlobDetectionModule,
    ParticleTrackingModule,
    XMeansClusteringModule,
    KMeansClusteringModule,
    DBSCANClusteringModule,
    ParticleRegionExtractionModule,
    ParticleRegionMediaExportModule,
    BoundarySizeModule,
    # FrameSegmenter,
    # etc.
]

BUILTIN_ANALYSIS_MODULES = {cls().name: cls for cls in _BUILTIN}
