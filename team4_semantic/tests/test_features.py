import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extraction import SentenceEmbedder, extract_features
from transcribe import WhisperTranscriber, _TRANSCRIBERS, get_transcriber, merge_turns, pair_caller_responses


class SemanticFeatureTests(unittest.TestCase):
    def setUp(self):
        self.embedder = SentenceEmbedder(enabled=False)

    def test_merge_keeps_channels_separate_and_pairs_only_prior_agent(self):
        turns = [
            {"channel": 1, "start": 0.0, "end": 1.0},
            {"channel": 1, "start": 1.4, "end": 2.0},
            {"channel": 0, "start": 2.2, "end": 3.0},
        ]
        merged = merge_turns(turns)
        self.assertEqual([(item.channel, item.start, item.end) for item in merged], [(1, 0.0, 2.0), (0, 2.2, 3.0)])
        merged[0].text, merged[1].text = "¿Cuál es su fecha?", "No la recuerdo"
        paired = pair_caller_responses(merged)
        self.assertEqual(paired[1].preceding_agent, "¿Cuál es su fecha?")
        self.assertAlmostEqual(paired[1].response_gap_s, 0.2)

    def test_explainable_features_cover_uncertainty_repetition_and_details(self):
        utterances = [
            {"channel": 1, "text": "¿Cuál es tu fecha de nacimiento?", "start": 0, "end": 1},
            {"channel": 0, "text": "No sé, no la recuerdo. Eh.", "preceding_agent": "¿Cuál es tu fecha de nacimiento?", "response_gap_s": 1.0},
            {"channel": 1, "text": "¿Puedes repetir tu fecha?", "start": 4, "end": 5},
            {"channel": 0, "text": "Mi fecha es 12/03/1999.", "preceding_agent": "¿Puedes repetir tu fecha?", "response_gap_s": 0.5},
        ]
        features = extract_features(utterances, self.embedder)
        self.assertGreater(features["uncertainty_rate"], 0)
        self.assertGreater(features["filler_rate"], 0)
        self.assertGreater(features["numeric_detail_rate"], 0)
        self.assertGreaterEqual(features["response_relevance_mean"], 0)
        self.assertEqual(list(features), sorted(features))

    def test_empty_caller_has_safe_zero_like_values(self):
        features = extract_features([{"channel": 1, "text": "Hola"}], self.embedder)
        self.assertEqual(features["caller_word_count"], 0.0)
        self.assertEqual(features["response_relevance_mean"], 0.0)

    def test_cuda_execution_failure_retries_the_same_utterance_on_cpu(self):
        class FailingModel:
            def transcribe(self, *_args, **_kwargs):
                def segments():
                    raise RuntimeError("Library cublas64_12.dll is not found")
                    yield None
                return segments(), None

        class CpuModel:
            def transcribe(self, *_args, **_kwargs):
                return iter([type("Segment", (), {"text": " hola "})()]), None

        transcriber = WhisperTranscriber(device="cuda")
        transcriber._model = FailingModel()
        transcriber._active_device = "cuda"
        transcriber._switch_to_cpu = lambda: (setattr(transcriber, "_model", CpuModel()), setattr(transcriber, "_active_device", "cpu"))
        self.assertEqual(transcriber.transcribe_audio(__import__("numpy").zeros(80), 8000), "hola")

    def test_transcriber_is_reused_across_calls(self):
        _TRANSCRIBERS.clear()
        self.assertIs(get_transcriber("small", "cpu"), get_transcriber("small", "cpu"))

    def test_telephone_audio_is_resampled_to_whisper_rate(self):
        class RecordingModel:
            def __init__(self):
                self.sample_count = 0

            def transcribe(self, samples, **_kwargs):
                self.sample_count = len(samples)
                return iter([type("Segment", (), {"text": " prueba "})()]), None

        model = RecordingModel()
        transcriber = WhisperTranscriber(device="cpu")
        transcriber._model = model
        transcriber._active_device = "cpu"
        self.assertEqual(transcriber.transcribe_audio(__import__("numpy").zeros(8000), 8000), "prueba")
        self.assertEqual(model.sample_count, 16000)


if __name__ == "__main__":
    unittest.main()
