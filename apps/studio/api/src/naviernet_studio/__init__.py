"""naviernet studio: the data service behind the post-processing studio.

The studio is coupled to the rest of the repository only through data: it reads
``data/<dataset>`` and ``outputs/<run>``, and evaluates a run's checkpoint through
the ``naviernet`` package itself. It computes no physics of its own; every number
it serves comes from a model accessor or a residual function the run trained
with, converted to physical units with the run's own scales.
"""
