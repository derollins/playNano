API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Core
   :name: core_api

   playnano
   playnano.afm_stack
   playnano.errors

.. toctree::
   :maxdepth: 2
   :caption: IO & Data Formats
   :name: io_api

   playnano.io
   playnano.io.loader
   playnano.io.data_loaders
   playnano.io.export_data
   playnano.io.gif_export
   playnano.io.formats
   playnano.io.formats.read_asd
   playnano.io.formats.read_h5jpk
   playnano.io.formats.read_jpk_folder
   playnano.io.formats.read_spm_folder

.. toctree::
   :maxdepth: 2
   :caption: Processing Pipeline
   :name: processing_api

   playnano.processing
   playnano.processing.core
   playnano.processing.filters
   playnano.processing.mask_generators
   playnano.processing.masked_filters
   playnano.processing.pipeline
   playnano.processing.stack_edit
   playnano.processing.video_processing

.. toctree::
   :maxdepth: 2
   :caption: Analysis & Modules
   :name: analysis_api

   playnano.analysis
   playnano.analysis.base
   playnano.analysis.pipeline
   playnano.analysis.export
   playnano.analysis.modules
   playnano.analysis.modules.count_nonzero
   playnano.analysis.modules.dbscan_clustering
   playnano.analysis.modules.feature_detection
   playnano.analysis.modules.k_means_clustering
   playnano.analysis.modules.log_blob_detection
   playnano.analysis.modules.particle_tracking
   playnano.analysis.modules.x_means_clustering
   playnano.analysis.utils
   playnano.analysis.utils.common
   playnano.analysis.utils.frames
   playnano.analysis.utils.loader
   playnano.analysis.utils.particles

.. toctree::
   :maxdepth: 2
   :caption: General Utilities
   :name: utils_api

   playnano.utils
   playnano.utils.colormaps
   playnano.utils.constants
   playnano.utils.io_utils
   playnano.utils.param_utils
   playnano.utils.system_info
   playnano.utils.time_utils
   playnano.utils.versioning

.. toctree::
   :maxdepth: 2
   :caption: CLI & App Utils
   :name: cli_api

   playnano.cli
   playnano.cli.actions
   playnano.cli.entrypoint
   playnano.cli.handlers
   playnano.cli.utils

.. toctree::
   :maxdepth: 2
   :caption: Graphical Interface
   :name: gui_api

   playnano.gui
   playnano.gui.main
   playnano.gui.window
   playnano.gui.widgets
   playnano.gui.widgets.controls
   playnano.gui.widgets.viewer
