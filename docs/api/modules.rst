API Reference
=============

.. toctree::
   :caption: Core
   :maxdepth: 2

   playnano


.. toctree::
   :caption: Analysis
   :maxdepth: 2

   playnano.analysis
   playnano.analysis.base
   playnano.analysis.export
   playnano.analysis.pipeline
   playnano.analysis.utils
   playnano.analysis.utils.common
   playnano.analysis.utils.frames
   playnano.analysis.utils.loader
   playnano.analysis.utils.particles


.. toctree::
   :caption: Analysis Modules
   :maxdepth: 2

   playnano.analysis.modules
   playnano.analysis.modules.count_nonzero
   playnano.analysis.modules.dbscan_clustering
   playnano.analysis.modules.feature_detection
   playnano.analysis.modules.k_means_clustering
   playnano.analysis.modules.log_blob_detection
   playnano.analysis.modules.particle_tracking
   playnano.analysis.modules.x_means_clustering


.. toctree::
   :caption: Processing
   :maxdepth: 2

   playnano.processing
   playnano.processing.core
   playnano.processing.filters
   playnano.processing.mask_generators
   playnano.processing.masked_filters
   playnano.processing.pipeline
   playnano.processing.stack_edit
   playnano.processing.video_processing


.. toctree::
   :caption: IO
   :maxdepth: 2

   playnano.io
   playnano.io.data_loaders
   playnano.io.export_data
   playnano.io.formats
   playnano.io.formats.read_asd
   playnano.io.formats.read_h5jpk
   playnano.io.formats.read_jpk_folder
   playnano.io.formats.read_spm_folder
   playnano.io.gif_export
   playnano.io.loader


.. toctree::
   :caption: CLI
   :maxdepth: 2

   playnano.cli
   playnano.cli.actions
   playnano.cli.entrypoint
   playnano.cli.handlers
   playnano.cli.utils


.. toctree::
   :caption: GUI
   :maxdepth: 2

   playnano.gui
   playnano.gui.main
   playnano.gui.widgets
   playnano.gui.widgets.controls
   playnano.gui.widgets.viewer
   playnano.gui.window