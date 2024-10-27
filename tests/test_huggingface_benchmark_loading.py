import unittest
from src.benchmarks.base_benchmark import HuggingfaceBenchmark, BenchmarkCategory, BenchmarkType
from dotenv import load_dotenv

class TestHuggingfaceBenchmark(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.benchmark_name = "ewok"
        self.benchmark_id = "ewok-core/ewok-core-1.0"
        self.type = BenchmarkType.FUNCTIONAL
        self.category = BenchmarkCategory.WORLD_KNOWLEDGE
        self.huggingface_benchmark = HuggingfaceBenchmark(name=self.benchmark_name,
                                                          type=self.type,
                                                          category=self.category,
                                                          description="",
                                                          id=self.benchmark_id,)


    def test_dataset_loading(self):
        # Test if the dataset loads successfully
        self.assertIsNotNone(self.huggingface_benchmark.benchmark)
        self.assertTrue(len(self.huggingface_benchmark.benchmark) > 0)


if __name__ == "__main__":
    unittest.main()