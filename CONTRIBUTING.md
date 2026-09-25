# Contributing

Install `.[dev]` in a virtual environment. Use a focused feature branch and include
an analytical or invariant-based check when changing numerical behavior. Run Ruff,
pytest, the example, `dampwatch --output out`, `python -m build`, and `pip check`
before merging. CI also reinstalls/tests the wheel on Windows and Linux.

Keep equations and alpha conventions explicit. Do not label a budget defect as
physical damping or as a monotone algorithmic energy. Regenerate sample outputs
together and explain numerical differences. Do not add private data or course
materials. Record only actual checks/reviews and keep meaningful development commits.
