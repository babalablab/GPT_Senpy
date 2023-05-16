import tiktoken
from typing import List


class Tokenizer:
    def __init__(self, model: str) -> None:
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a text into a list of tokens.

        Args:
            text: str

        Returns:
            List[str]

        Raises:
            ValueError: If input text is not string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        return self._decode(self._encode(text))

    def count_tokens(self, text: str) -> int:
        """
        Count tokens after tokenized in a text.

        Args:
            text: str

        Returns:
            int

        Raises:
            ValueError: If input text is not string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        tokens = self._encode(text)
        return len(tokens)

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def _decode(self, token: List[int]) -> List[str]:
        return self.tokenizer.decode(token)
