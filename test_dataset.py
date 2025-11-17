import unittest
import torch
from torch.utils.data import DataLoader
from dataset import WeeklySentimentDataset, collate_weekly, demo_random_dataset

class TestWeeklySentimentDataset(unittest.TestCase):
    def test_basic_init_and_getitem(self):
        ds, meta = demo_random_dataset(n=4, seq_len=5, dim=16)
        self.assertEqual(len(ds), 4)
        item = ds[0]
        self.assertIn('x', item)
        self.assertIn('length', item)
        self.assertTrue(item['x'].shape[1] == 16)

    def test_collate(self):
        ds, _ = demo_random_dataset(n=3, seq_len=5, dim=8)
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_weekly)
        batch = next(iter(loader))
        self.assertEqual(batch['x'].dim(), 3)  # [B, T, D]
        self.assertEqual(batch['x'].shape[0], 3)
        self.assertEqual(batch['lengths'].shape[0], 3)
        self.assertIn('direction', batch)
        self.assertEqual(batch['direction'].shape[0], 3)

if __name__ == '__main__':
    unittest.main()
