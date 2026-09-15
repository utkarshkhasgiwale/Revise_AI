"""Evaluation system (re-exported from ytrag.evaluate for backward compatibility)."""

from ytrag.evaluate import DEFAULT_GOLDEN, compare_retrievers, load_golden, run_eval

__all__ = ["DEFAULT_GOLDEN", "compare_retrievers", "load_golden", "run_eval"]
