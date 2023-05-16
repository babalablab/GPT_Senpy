import unittest

from gptsenpy.Tokenizer import Tokenizer


class TestTokenizerMethods(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-4"
        self.text = "Hello, my name is John"

    def test_count_tokenizer(self):
        tokenizer = Tokenizer(self.model)
        n = tokenizer.count_tokens(self.text)
        self.assertGreater(n, 0)

    def test_error_count_tokenizer(self):
        tokenizer = Tokenizer(self.model)
        with self.assertRaises(ValueError):
            _ = tokenizer.count_tokens(10000000)


if __name__ == "__main__":
    unittest.main()
