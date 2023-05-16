import unittest

from gptsenpy.Tokenizer import Tokenizer


class TestTokenizerMethods(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-4"
        self.text = "Hello, my name is John"
        self.tokenizer = Tokenizer(self.model)

    def test_tokenizer(self):
        tokens = self.tokenizer.tokenize(self.text)
        self.assertGreater(len(tokens), 0)

    def test_count_tokenizer(self):
        n = self.tokenizer.count_tokens(self.text)
        self.assertGreater(n, 0)

    def test_error_count_tokenizer(self):
        with self.assertRaises(ValueError):
            _ = self.tokenizer.count_tokens(10000000)


if __name__ == "__main__":
    unittest.main()
