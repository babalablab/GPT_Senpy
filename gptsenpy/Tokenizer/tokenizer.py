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

    def divide_text_by_max_token(self, text: str, max_tokens: int = 4000) -> list[str]:
        """
        This function divides a given text into smaller chunks based on a maximum number of tokens.
        The function first tokenizes the input text, then divides the tokens into chunks,
        each containing no more than `max_tokens` number of tokens.
        These chunks are then decoded back into text and returned as a list of strings.

        Args:
            text (str): The input text to be divided.
            max_tokens (int, optional): The maximum number of tokens for each divided text.
                                         Defaults to 4000.

        Returns:
            list[str]: A list of divided texts, each contains no more than `max_tokens` tokens.
        """
        tokens = self._encode(text)
        divided_tokens = []
        if len(tokens) > max_tokens:
            for i in range(0, len(tokens), max_tokens):
                divided_tokens.append(tokens[i : i + max_tokens])
        else:
            divided_tokens = [tokens]
        divided_texts = [self._decode(token) for token in divided_tokens]
        return divided_texts

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode(self, token: list[int]) -> str:
        return self.tokenizer.decode(token)
