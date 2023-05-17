import sys

sys.path.append("..")
sys.path.append("../gptsenpy")

from pytest import raises, warns

from gptsenpy.Prompt import PromptTemplate


def test_build_prompt_valid_template():
    # Valid template and replacer
    template = "Hello, my name is ${name} and I am ${age} years old."
    replacer = {"name": "John", "age": "30"}
    PT = PromptTemplate(template)
    expected_output = "Hello, my name is John and I am 30 years old."
    assert PT.build_prompt(replacer) == expected_output


def test_build_prompt_invalid_template():
    # Invalid template with missing placeholder
    template = "Hello, my name is ${name} and I am ${age} years old."
    replacer = {"name": "John"}
    PT = PromptTemplate(template)
    with raises(KeyError):
        PT.build_prompt(replacer)


def test_build_prompt_empty_replacer():
    # Empty replacer dictionary
    template = "Hello, my name is ${name} and I am ${age} years old."
    replacer = {}
    PT = PromptTemplate(template)
    with raises(KeyError):
        PT.build_prompt(replacer)


def test_build_prompt_empty_template():
    # Empty template
    template = ""
    replacer = {"name": "John", "age": "30"}
    # waring
    with warns(UserWarning):
        PT = PromptTemplate(template)
    expected_output = ""
    assert PT.build_prompt(replacer) == expected_output


def test_template_getter():
    # Test getter for template property
    template = "Hello, ${name}!"
    PT = PromptTemplate(template)
    assert PT.template == template


def test_template_setter():
    # Test setter for template property
    template = "Hello, ${name}!"
    new_template = "Hi, ${name}!"
    PT = PromptTemplate(template)
    PT.template = new_template
    assert PT.template == new_template


def test_template_empty():
    # Test template property with empty value
    empty_template = ""
    PT = PromptTemplate(empty_template)
    assert PT.template == empty_template


def test_template_not_equal():
    # Test template property with different values
    template = "Hello, ${name}!"
    different_template = "Hi, ${name}!"
    PT = PromptTemplate(template)
    assert PT.template != different_template


if __name__ == "__main__":
    pass
