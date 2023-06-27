import re
from pathlib import Path

import fitz


class PDFLoader:
    """
    The PDFLoader class provides functionality to load a PDF file and extract its text data.

    Attributes:
        path (Path): The path of the PDF file.
        doc (fitz.fitz.Document): The PDF document object.
        page_count (int): The number of pages in the PDF document.
    """

    def __init__(self, path: Path | str):
        """
        Initializes the PDFLoader with the provided path.

        Args:
            path (Path | str): The path of the PDF file.
        """
        self.path = Path(path)
        self.doc: fitz.fitz.Document = fitz.open(self.path)
        self.page_count = self.doc.page_count

    def get_page_text(self, pno: int = -1) -> str:
        """
        Extracts and returns the text from a specific page in the PDF document.

        Args:
            pno (int): The page number from which to extract the text.
                                 If -1 or greater than the total page count, the function extracts
                                 text from all pages. Defaults to -1.

        Returns:
            str: The extracted text.
        """
        if pno == -1 or pno >= self.page_count:
            _texts = [
                (" ".join(l.get_text().splitlines())).replace("- ", "")
                for l in self.doc
            ]
            texts = " ".join(_texts)
            return texts
        else:
            return (" ".join(self.doc.get_page_text(pno).splitlines())).replace(
                "- ", ""
            )

    def split_sections(
        self,
        pattern: str = r"(?<=\.\s)(?=\d+\.\s?[A-Z])",
    ) -> list[str]:
        """
        Splits the text from the PDF document into sections based on a regex pattern.

        Args:
            pattern (str): The regex pattern used for splitting the text into sections.
                                      Defaults to r"(?<=\.\s)(?=\d+\.\s?[A-Z])".

        Returns:
            list[str]: A list of text sections.
        """
        text = self.get_page_text(-1)
        sections = re.split(pattern, text)
        sections = [section for section in sections if section.strip()]
        return sections
