import unittest
import importlib.util
import os

### TODO: Update this after fine-tuning script is complete. 
class TestFineTuningScript(unittest.TestCase):
    def test_import_and_model_loading(self):
        script_path = os.path.join(os.path.dirname(__file__), 'fine-tuning.py')
        spec = importlib.util.spec_from_file_location("fine_tuning", script_path)
        fine_tuning = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(fine_tuning)
        except Exception as e:
            self.fail(f"Script failed to run: {e}")
        # Check that tokenizer and model are loaded
        self.assertIsNotNone(fine_tuning.tokenizer)
        self.assertIsNotNone(fine_tuning.model)

if __name__ == "__main__":
    unittest.main()
