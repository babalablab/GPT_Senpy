import re
from pathlib import Path
from typing import Optional

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
        if pno <= 0 or pno > self.page_count:
            _texts = [
                (" ".join(page.get_text().splitlines())).replace("- ", "")
                for page in self.doc
            ]
            texts = " ".join(_texts)
            return texts
        else:
            return (" ".join(self.doc.get_page_text(pno - 1).splitlines())).replace(
                "- ", ""
            )

    def get_page_text_by_range(
        self, min_pno: Optional[int], max_pno: Optional[int]
    ) -> str:
        """
        This method extracts and concatenates text from a range of pages in a document.

        Parameters:
        min_pno (int, optional): The starting page number from where to extract the text. The numbering starts from 1. Default is 1.
        max_pno (int, optional): The ending page number up to where to extract the text. If None, it defaults to the total number of pages in the document.

        Returns:
        str: A string of concatenated text from the specified range of pages. Text from each page is separated by a space. Line breaks within the text are replaced with spaces, and hyphenated line breaks are removed.

        Raises:
        AssertionError: If the provided page range is invalid. The valid page range is from 1 to the total number of pages in the document. The min_pno should be less than or equal to max_pno.

        Example:
        get_page_text_by_range(1, 2) would return the text from the first page up to but not including the second page.
        """
        min_pno = 1 if max_pno is None else min_pno
        max_pno = self.page_count if max_pno is None else max_pno
        assert min_pno is not None and max_pno is not None

        min_pno, max_pno = min(min_pno, max_pno), max(min_pno, max_pno)
        assert 1 <= min_pno <= max_pno <= self.page_count, "Invalid page range"

        _texts = []

        for pno in range(min_pno - 1, max_pno):
            _texts.append(
                (" ".join(self.doc.get_page_text(pno).splitlines())).replace("- ", "")
            )

        texts = " ".join(_texts)
        return texts

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
        sections = [section.strip() for section in sections]
        return sections
