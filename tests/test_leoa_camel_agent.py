"""Verify the camelCase validator without contacting any model service."""

import pytest

from agents590.leoa_CamelAgent import problems

ORIGINAL = """\
import json


def pay(sub_total):
    tax_rate = 0.08
    return json.dumps({"sub_total": sub_total * tax_rate}, sort_keys=True)
"""
CONVERTED = """\
import json


def pay(subTotal):
    taxRate = 0.08
    return json.dumps({"sub_total": subTotal * taxRate}, sort_keys=True)
"""


def test_correct_conversion_is_accepted() -> None:
    assert problems(ORIGINAL, CONVERTED) == ""


@pytest.mark.parametrize(
    ("converted", "expected"),
    [
        (ORIGINAL, "still use snake_case"),
        (CONVERTED.replace("taxRate = 0.08", "tax_rate = 0.08"), "still use snake_case"),
        (CONVERTED.replace('"sub_total"', '"subTotal"'), "changed"),
        (CONVERTED.replace("sort_keys", "sortKeys"), "changed"),
        (CONVERTED.replace("0.08", "0.09"), "changed"),
        ("def pay(subTotal:\n", "Syntax error"),
    ],
)
def test_bad_conversions_are_rejected(converted: str, expected: str) -> None:
    assert expected in problems(ORIGINAL, converted)
