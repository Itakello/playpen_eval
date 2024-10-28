import unittest

from dotenv import load_dotenv

from src.models.huggingface_model import HuggingfaceModel


class TestHuggingfaceModel(unittest.TestCase):

    def setUp(self):
        self.model_name = "google/gemma-2b-it"
        load_dotenv()

    def test_model_initialization(self):
        model = HuggingfaceModel(name=self.model_name)

        # Test if the model is initialized correctly
        self.assertIsNotNone(model)
        self.assertEqual(model.name, self.model_name)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.tokenizer)

        print(f"\nModel '{self.model_name}' initialized successfully.")

    def test_text_generation(self):
        model = HuggingfaceModel(name=self.model_name)

        prompt = "Explain the concept of machine learning in one sentence:"
        generated_text = model.generate(prompt)

        # Check if generated text is not empty and is a string
        self.assertIsInstance(generated_text, str)
        self.assertTrue(len(generated_text) > 0)

        # Print the generated text for manual inspection
        print(f"\nPrompt: {prompt}")
        print(f"Generated text:\n{generated_text}")


if __name__ == "__main__":
    unittest.main()
