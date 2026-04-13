"""Integration tests for PipelineRunner subclasses.

These smoke tests verify that each PipelineRunner produces structurally valid
SearchResult objects when given synthetic data. No file I/O, real training,
or model checkpoints are required.
"""

import logging

logger = logging.getLogger(__name__)
