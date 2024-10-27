import unittest
from src.benchmarks.base_benchmark import BenchmarkCategory, BenchmarkType
from src.benchmarks.ewok_benchmark import EwokBenchmark
from dotenv import load_dotenv

class TestEwokBenchmark(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.benchmark_name = "ewok"
        self.type = BenchmarkType.FUNCTIONAL
        self.category = BenchmarkCategory.WORLD_KNOWLEDGE
        self.huggingface_benchmark = EwokBenchmark(name=self.benchmark_name,
                                                          type=self.type,
                                                          category=self.category,
                                                          description="")


    def test_dataset_loading(self):
        # Test if the dataset loads successfully
        self.assertIsNotNone(self.huggingface_benchmark.benchmark)
        self.assertTrue(len(self.huggingface_benchmark.benchmark) > 0)


if __name__ == "__main__":
    unittest.main()