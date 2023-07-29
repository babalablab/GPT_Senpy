import warnings
from string import Template


class PromptTemplate:
    def __init__(self, template: str) -> None:
        if not template:
            warnings.warn("Template is empty!!")
        self.__template: str = template

    @property
    def template(self):
        return self.__template

    @template.setter
    def template(self, template):
        self.__template = template

    def build_prompt(self, replacer: dict) -> str:
        """
        Builds a prompt string by replacing placeholders in a template string with values from a dictionary.

        Args:
            replacer (dict): A dictionary containing key-value pairs where the keys are placeholders in the template string and the values are the values to replace them with.

        Returns:
            str: The prompt string with all placeholders replaced with their corresponding values.

        Raises:
            KeyError: If replacer does not have enough keys.
            ValueError: If the template string is invalid and cannot be substituted.

        Example:
            >>> replacer = {'name': 'John', 'age': '30'}
            >>> template = 'Hello, my name is ${name} and I am ${age} years old.'
            >>> prompt_template = PromptTemplate(template)
            >>> prompt_template.build_prompt(replacer)
            'Hello, my name is John and I am 30 years old.'
        """

        template = Template(self.__template)

        try:
            return template.substitute(**replacer)
        except KeyError as e:
            raise KeyError(f"Invalid template: Missing placeholder '{e.args[0]}'")
        except ValueError:
            raise ValueError("Invalid template")
