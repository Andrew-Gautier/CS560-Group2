import unittest
import torch
from model import LSTMClassifier, test_model


class TestLSTMClassifier(unittest.TestCase):
    def test_forward_shapes(self):
        shapes = test_model()  # uses internal random test
        self.assertEqual(tuple(shapes["direction_logit"]), (4, 1))
        self.assertEqual(tuple(shapes["attn_weights"]), (4, 5))

    def test_variable_lengths(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMClassifier(input_dim=768).to(device)
        x = torch.randn(2, 6, 768, device=device)
        lengths = torch.tensor([6, 3], device=device)
        out = model(x, lengths)

        # Attention on padded steps should be exactly 0 (due to masking with -inf before softmax)
        padded_slice = out["attn_weights"][1, 3:]
        padded_count = int(torch.count_nonzero(padded_slice).item())
        self.assertEqual(padded_count, 0)

        # Basic probability bounds
        probs = out["direction_prob"]
        self.assertTrue(torch.all(probs >= 0).item() and torch.all(probs <= 1).item())


if __name__ == "__main__":
    unittest.main()
