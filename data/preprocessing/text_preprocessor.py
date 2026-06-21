"""
Text Preprocessor for DG-HMCF.

Tokenises participant transcripts with RoBERTa and extracts a set of
psycholinguistic features known to correlate with depression.
"""

import re
import string
from typing import Dict, List, Optional

import numpy as np

try:
    from transformers import RobertaTokenizerFast
except ImportError:
    RobertaTokenizerFast = None  # type: ignore

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None  # type: ignore

# ---------------------------------------------------------------------------
# Word lists for linguistic feature computation
# ---------------------------------------------------------------------------

NEGATIVE_WORDS: List[str] = [
    "sad", "depressed", "hopeless", "worthless", "empty", "miserable",
    "terrible", "horrible", "awful", "dreadful", "hate", "lonely",
    "useless", "failure", "tired", "exhausted", "numb", "pain", "hurt",
    "stuck", "trapped", "lost", "alone", "dark", "heavy", "cry", "crying",
    "worried", "anxious", "afraid", "scared", "helpless",
]

HEDGE_WORDS: List[str] = [
    "maybe", "perhaps", "possibly", "might", "could", "sometimes",
    "kind of", "sort of", "i think", "i guess", "i suppose", "not sure",
    "uncertain", "unsure", "unclear", "probably", "usually", "often",
    "sometimes", "rarely", "hardly", "barely",
]

FIRST_PERSON_PRONOUNS: List[str] = [
    "i", "me", "my", "myself", "mine",
]

EMOTIONAL_POSITIVE_WORDS: List[str] = [
    "happy", "good", "great", "wonderful", "joyful", "excited", "love",
    "enjoy", "positive", "hope", "better", "improve", "smile", "laugh",
]

EMOTIONAL_NEGATIVE_WORDS: List[str] = NEGATIVE_WORDS


class TextPreprocessor:
    """
    Preprocesses interview transcript text for depression detection.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the RoBERTa tokeniser.
    max_length : int
        Maximum token sequence length for the tokeniser.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length

        if RobertaTokenizerFast is not None:
            try:
                self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        transcript: str,
        remove_interviewer: bool = True,
    ) -> Dict:
        """
        Tokenise a raw transcript and extract linguistic features.

        Parameters
        ----------
        transcript : str
            Raw interview transcript text.
        remove_interviewer : bool
            If True, strip lines beginning with "Ellie:" or "Interviewer:".

        Returns
        -------
        dict with keys:
            ``input_ids``          – np.ndarray, shape (max_length,)
            ``attention_mask``     – np.ndarray, shape (max_length,)
            ``linguistic_features``– np.ndarray, shape (5,)
        """
        if remove_interviewer:
            transcript = self._remove_interviewer_lines(transcript)

        participant_text = transcript.strip()

        # Tokenise
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                participant_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            input_ids = encoding["input_ids"][0].astype(np.int64)
            attention_mask = encoding["attention_mask"][0].astype(np.float32)
        else:
            # Fallback: dummy tokens
            input_ids = np.zeros(self.max_length, dtype=np.int64)
            attention_mask = np.zeros(self.max_length, dtype=np.float32)

        linguistic_features = self.extract_linguistic_features(participant_text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "linguistic_features": linguistic_features.astype(np.float32),
        }

    def extract_linguistic_features(self, text: str) -> np.ndarray:
        """
        Compute a 5-dimensional psycholinguistic feature vector.

        Returns
        -------
        np.ndarray, shape (5,)
            [sentiment_score, neg_word_ratio, uncertainty_score,
             self_reference_freq, emotional_polarity]
        """
        sentiment = self._compute_sentiment(text)
        neg_word_ratio = self._compute_neg_word_ratio(text)
        uncertainty = self._compute_uncertainty_score(text)
        self_ref = self._compute_self_reference(text)
        emotional_polarity = self._compute_emotional_polarity(text)

        features = np.array(
            [sentiment, neg_word_ratio, uncertainty, self_ref, emotional_polarity],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _compute_sentiment(self, text: str) -> float:
        """
        Overall sentiment polarity in [-1, 1] using TextBlob.
        Falls back to lexicon-based scoring if TextBlob is unavailable.
        """
        if TextBlob is not None and text.strip():
            try:
                return float(TextBlob(text).sentiment.polarity)
            except Exception:
                pass
        # Fallback: simple positive/negative word ratio
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in EMOTIONAL_POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in EMOTIONAL_NEGATIVE_WORDS)
        total = len(tokens)
        return float((pos - neg) / total)

    def _compute_neg_word_ratio(self, text: str) -> float:
        """
        Ratio of negative words to total word count.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        return float(neg_count / len(tokens))

    def _compute_uncertainty_score(self, text: str) -> float:
        """
        Frequency of hedge/uncertainty words per 100 words.
        """
        text_lower = text.lower()
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        hedge_count = 0
        for hedge in HEDGE_WORDS:
            # Use phrase matching for multi-word hedges
            hedge_count += len(re.findall(r"\b" + re.escape(hedge) + r"\b", text_lower))
        return float(hedge_count / len(tokens))

    def _compute_self_reference(self, text: str) -> float:
        """
        First-person pronoun frequency per 100 words.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        first_person_count = sum(1 for t in tokens if t in FIRST_PERSON_PRONOUNS)
        return float(first_person_count / len(tokens))

    def _compute_emotional_polarity(self, text: str) -> float:
        """
        Ratio of positive to negative emotional words.
        Scaled to [-1, 1]: -1 = purely negative, +1 = purely positive.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        pos = sum(1 for t in tokens if t in EMOTIONAL_POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in EMOTIONAL_NEGATIVE_WORDS)
        total_emotional = pos + neg
        if total_emotional == 0:
            return 0.0
        return float((pos - neg) / total_emotional)

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_interviewer_lines(transcript: str) -> str:
        """
        Remove lines that start with interviewer speaker labels.
        Handles 'Ellie:', 'Interviewer:', 'E:', 'INT:' prefixes.
        """
        pattern = re.compile(
            r"^(ellie|interviewer|int|e)\s*:.*$",
            re.IGNORECASE | re.MULTILINE,
        )
        cleaned = pattern.sub("", transcript)
        # Also remove participant speaker labels like "P:", "Participant:"
        participant_label = re.compile(
            r"^(participant|p)\s*:\s*",
            re.IGNORECASE | re.MULTILINE,
        )
        cleaned = participant_label.sub("", cleaned)
        # Collapse multiple blank lines
        cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [t for t in text.split() if t]
