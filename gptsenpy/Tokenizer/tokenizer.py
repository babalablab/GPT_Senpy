import tiktoken


class Tokenizer:
    def __init__(self, model: str) -> None:
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str) -> list[str]:
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
        list_tokens: list[int] = self._encode(text)
        _tokens: list = [
            self.tokenizer.decode_single_token_bytes(token) for token in list_tokens
        ]
        tokens: list = [token.decode("utf-8") for token in _tokens]
        return tokens

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

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode(self, token: list[int]) -> str:
        return self.tokenizer.decode(token)
