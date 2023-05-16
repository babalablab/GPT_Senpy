import unittest

from gptsenpy.Prompt import PromptTemplate


class TestPromptMethods(unittest.TestCase):
    def setUp(self):
        self.templates = [
            "${text}, ${author}, ${title}",
        ]
        self.replacers = [
            {"text": "te", "author": "au", "title": "ti"},
        ]

    def test_build_prompt_0(self):
        PT = PromptTemplate(self.templates[0])
        ans = "te, au, ti"
        self.assertEqual(PT.build_prompt(self.replacers[0]), ans)

    def test_build_prompt_valid_template(self):
        # Valid template and replacer
        template = "Hello, my name is ${name} and I am ${age} years old."
        replacer = {"name": "John", "age": "30"}
        PT = PromptTemplate(template)
        expected_output = "Hello, my name is John and I am 30 years old."
        self.assertEqual(PT.build_prompt(replacer), expected_output)

    def test_build_prompt_invalid_template(self):
        # Invalid template with missing placeholder
        template = "Hello, my name is ${name} and I am ${age} years old."
        replacer = {"name": "John"}
        PT = PromptTemplate(template)
        with self.assertRaises(KeyError):
            PT.build_prompt(replacer)

    def test_build_prompt_empty_replacer(self):
        # Empty replacer dictionary
        template = "Hello, my name is ${name} and I am ${age} years old."
        replacer = {}
        PT = PromptTemplate(template)
        with self.assertRaises(KeyError):
            PT.build_prompt(replacer)

    def test_build_prompt_empty_template(self):
        # Empty template
        template = ""
        replacer = {"name": "John", "age": "30"}
        # waring
        with self.assertWarns(UserWarning):
            PT = PromptTemplate(template)
        expected_output = ""
        self.assertEqual(PT.build_prompt(replacer), expected_output)

    def test_template_getter(self):
        # Test getter for template property
        template = "Hello, ${name}!"
        PT = PromptTemplate(template)
        self.assertEqual(PT.template, template)

    def test_template_setter(self):
        # Test setter for template property
        template = "Hello, ${name}!"
        new_template = "Hi, ${name}!"
        PT = PromptTemplate(template)
        PT.template = new_template
        self.assertEqual(PT.template, new_template)

    def test_template_empty(self):
        # Test template property with empty value
        empty_template = ""
        PT = PromptTemplate(empty_template)
        self.assertEqual(PT.template, empty_template)

    def test_template_not_equal(self):
        # Test template property with different values
        template = "Hello, ${name}!"
        different_template = "Hi, ${name}!"
        PT = PromptTemplate(template)
        self.assertNotEqual(PT.template, different_template)


if __name__ == "__main__":
    unittest.main()
