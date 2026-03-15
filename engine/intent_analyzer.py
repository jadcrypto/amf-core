"""
Intent Analyzer (محلل النية)
============================
Lightweight classifier that determines which functional cells
a query needs, triggering pre-fetching of relevant weight cells.

Pipeline:
    User Prompt → Vectorization → Pattern Matching → Cell Selection

Features:
- Probabilistic intent detection with confidence scoring
- Compound intent handling (e.g., "explain code poetically" → logic + creative)
- Refinement: narrows cell selection as more tokens are processed
- Pre-fetching triggers when confidence > threshold
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Result of intent analysis."""
    primary_intent: str               # Dominant intent category
    confidence: float                 # 0.0 to 1.0
    intent_scores: dict = field(default_factory=dict)  # All intents with scores
    required_zones: list = field(default_factory=list)  # Layer zones needed
    required_groups: list = field(default_factory=list)  # Functional groups needed
    is_compound: bool = False         # Multi-intent query
    compound_intents: list = field(default_factory=list)

    @property
    def should_prefetch(self) -> bool:
        """Whether confidence is high enough to trigger pre-fetching."""
        from config import INTENT_CONFIDENCE_THRESHOLD
        return self.confidence >= INTENT_CONFIDENCE_THRESHOLD


# ============================================================
# Intent Library — keyword patterns and zone mappings
# ============================================================

INTENT_PATTERNS = {
    "math_logic": {
        "keywords": [
            "calculate", "compute", "solve", "equation", "math",
            "number", "formula", "algorithm", "proof", "theorem",
            "sum", "product", "integral", "derivative", "probability",
            "percentage", "average", "median", "statistics",
            "احسب", "حل", "معادلة", "رياضيات", "جمع", "ضرب",
            "نسبة", "إحصاء", "برهان",
        ],
        "zones": ["reasoning", "semantic"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 1.0,
    },
    "code_programming": {
        "keywords": [
            "code", "program", "function", "class", "variable",
            "debug", "error", "python", "javascript", "html",
            "api", "database", "sql", "git", "compile", "run",
            "script", "loop", "array", "string", "import",
            "برمجة", "كود", "دالة", "متغير", "خطأ", "تصحيح",
        ],
        "zones": ["reasoning", "semantic"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 1.0,
    },
    "language_grammar": {
        "keywords": [
            "grammar", "spelling", "correct", "sentence", "word",
            "verb", "noun", "adjective", "syntax", "punctuation",
            "paragraph", "essay", "rewrite", "edit", "proofread",
            "قواعد", "إملاء", "صحح", "جملة", "كلمة", "فعل", "اسم",
            "نحو", "صرف",
        ],
        "zones": ["linguistic", "semantic"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 1.0,
    },
    "creative_writing": {
        "keywords": [
            "write", "story", "poem", "creative", "imagine",
            "fiction", "character", "plot", "describe", "narrative",
            "metaphor", "lyric", "song", "artistic", "compose",
            "اكتب", "قصة", "شعر", "إبداع", "تخيل", "رواية",
            "وصف", "أغنية",
        ],
        "zones": ["linguistic", "semantic", "reasoning"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 0.9,
    },
    "general_knowledge": {
        "keywords": [
            "what", "who", "when", "where", "why", "how",
            "explain", "define", "describe", "tell", "about",
            "history", "science", "geography", "fact",
            "ما", "من", "متى", "أين", "لماذا", "كيف",
            "اشرح", "عرف", "صف", "أخبرني", "تاريخ", "علم",
        ],
        "zones": ["semantic", "reasoning"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 0.8,
    },
    "translation": {
        "keywords": [
            "translate", "translation", "convert", "language",
            "english", "arabic", "french", "spanish",
            "ترجم", "ترجمة", "حول", "إنجليزي", "عربي", "فرنسي",
        ],
        "zones": ["linguistic", "semantic"],
        "groups": ["ATTENTION", "FFN"],
        "weight": 1.0,
    },
}

# Zone → intent affinity mapping
ZONE_INTENT_AFFINITY = {
    "linguistic": ["language_grammar", "translation", "creative_writing"],
    "semantic": ["general_knowledge", "translation", "creative_writing"],
    "reasoning": ["math_logic", "code_programming"],
}


class IntentAnalyzer:
    """
    Analyzes user prompts to determine required functional cells.

    Uses keyword-based pattern matching with TF-IDF-style weighting
    for fast, lightweight classification without external models.
    """

    def __init__(self):
        self._patterns = INTENT_PATTERNS
        self._history: list[IntentResult] = []

    def analyze(self, prompt: str) -> IntentResult:
        """
        Analyze a prompt and determine required cells.

        Args:
            prompt: User's input text.

        Returns:
            IntentResult with cell requirements.
        """
        prompt_lower = prompt.lower().strip()
        tokens = self._tokenize(prompt_lower)

        # Score each intent category
        scores = {}
        for intent_name, pattern in self._patterns.items():
            score = self._compute_score(tokens, pattern)
            if score > 0:
                scores[intent_name] = score

        # Normalize scores
        total = sum(scores.values()) if scores else 1.0
        scores = {k: v / total for k, v in scores.items()}

        # Determine primary intent
        if scores:
            primary = max(scores, key=scores.get)
            confidence = scores[primary]
        else:
            primary = "general_knowledge"
            confidence = 0.5
            scores = {"general_knowledge": 0.5}

        # Check for compound intents
        is_compound = False
        compound = []
        significant = [
            (k, v) for k, v in scores.items()
            if v > 0.2 and k != primary
        ]
        if significant:
            is_compound = True
            compound = [primary] + [k for k, v in significant]

        # Determine required zones and groups
        required_zones = set()
        required_groups = set(["ATTENTION", "FFN"])  # Always need both

        for intent_name in ([primary] + [k for k, _ in significant]):
            pattern = self._patterns.get(intent_name, {})
            for zone in pattern.get("zones", []):
                required_zones.add(zone)

        # Always include core
        required_zones = list(required_zones) if required_zones else [
            "semantic", "reasoning"
        ]

        result = IntentResult(
            primary_intent=primary,
            confidence=confidence,
            intent_scores=scores,
            required_zones=required_zones,
            required_groups=list(required_groups),
            is_compound=is_compound,
            compound_intents=compound,
        )

        self._history.append(result)
        logger.debug(
            f"Intent: {primary} (conf={confidence:.2f}), "
            f"zones={required_zones}"
        )
        return result

    def refine(self, result: IntentResult, new_tokens: list[str]) -> IntentResult:
        """
        Refine intent prediction with additional tokens.
        Narrows cell selection as more context becomes available.
        """
        # Re-score with additional tokens
        scores = dict(result.intent_scores)

        for intent_name, pattern in self._patterns.items():
            additional_score = 0
            for token in new_tokens:
                if token in pattern["keywords"]:
                    additional_score += pattern["weight"] * 0.5

            if additional_score > 0:
                scores[intent_name] = scores.get(intent_name, 0) + additional_score

        # Re-normalize
        total = sum(scores.values()) if scores else 1.0
        scores = {k: v / total for k, v in scores.items()}

        primary = max(scores, key=scores.get)
        result.primary_intent = primary
        result.confidence = scores[primary]
        result.intent_scores = scores

        return result

    def get_required_cell_ids(self, result: IntentResult) -> list[str]:
        """
        Map an IntentResult to specific cell IDs.
        Returns cell IDs that the Molecular Engine should load.
        """
        cell_ids = ["core"]  # Always need core

        for zone in result.required_zones:
            for group in result.required_groups:
                if group == "ATTENTION":
                    cell_ids.append(f"{zone}_attn")
                elif group == "FFN":
                    cell_ids.append(f"{zone}_ffn")

        return list(set(cell_ids))

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        # Remove punctuation, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    def _compute_score(self, tokens: list[str], pattern: dict) -> float:
        """Compute match score between tokens and a pattern."""
        keywords = pattern["keywords"]
        weight = pattern["weight"]
        matches = 0

        for token in tokens:
            for keyword in keywords:
                if token == keyword or keyword in token:
                    matches += 1
                    break

        if not tokens:
            return 0.0

        return (matches / len(tokens)) * weight
