"""PolicyInduction: boosted, interpretable classification with LLM-written rules
scored by TypeSafe Jev."""

from .config import BoostConfig, WeightConfig
from .generator import GoogleGenerator, OpenAIGenerator, RuleGenerator, make_generator
from .model import PolicyInduction
from .scorer import JevScorer, Scorer

__all__ = [
    "PolicyInduction",
    "WeightConfig",
    "BoostConfig",
    "JevScorer",
    "Scorer",
    "RuleGenerator",
    "OpenAIGenerator",
    "GoogleGenerator",
    "make_generator",
]
