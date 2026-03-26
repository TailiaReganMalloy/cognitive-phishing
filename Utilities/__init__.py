"""Convenience exports for the Utilities package."""

try:
	from .CognitivePhishingEvaluator import CognitivePhishingEvaluator
except ModuleNotFoundError:
	# Optional dependency: evaluator requires ollama, which is not needed for all workflows.
	CognitivePhishingEvaluator = None
from .CognitivePhishingRAG import CognitivePhishingRAG

try:
	from .Utilities import *
	from .Utilities import __all__ as _utils_all
except ModuleNotFoundError:
	# Optional dependency: legacy utility module requires vertexai.
	_utils_all = []

__all__ = [
	"CognitivePhishingEvaluator",
	"CognitivePhishingRAG",
	*_utils_all,
]