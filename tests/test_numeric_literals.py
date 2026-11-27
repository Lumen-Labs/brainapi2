import unittest

from src.core.saving.identity import stable_relationship_id
from src.core.saving.numeric_literals import (
    amount_identity_token,
    amounts_differ,
    display_numeric_value,
    parse_stated_number,
    retain_numeric_on_relationship,
)


class _Endpoint:
    def __init__(self, name, typ="EVENT"):
        self.name = name
        self.type = typ
        self.description = ""
        self.labels = [typ]


class _Rel:
    def __init__(
        self,
        *,
        amount=None,
        description="",
        name="FOUNDED_IN",
        tail=None,
        tip=None,
        source_span="",
        properties=None,
    ):
        self.amount = amount
        self.description = description
        self.name = name
        self.tail = tail or _Endpoint("Halvorsen Dynamics", "ORGANIZATION")
        self.tip = tip or _Endpoint("Halvorsen Dynamics founding", "EVENT")
        self.source_span = source_span
        self.properties = dict(properties or {})


class NumericLiteralTests(unittest.TestCase):
    def test_parse_years_and_comma_prices(self):
        self.assertEqual(parse_stated_number("Founded in 2011"), 2011.0)
        self.assertEqual(parse_stated_number("price 12,400"), 12400.0)
        self.assertEqual(parse_stated_number("14,900"), 14900.0)
        self.assertEqual(parse_stated_number("5.20"), 5.2)

    def test_parse_skips_model_tokens_and_prefers_price(self):
        cheap = parse_stated_number("Kestrel X2 pump costs 12,400 EUR")
        dear = parse_stated_number("Kestrel X2 pump costs 14,900 EUR")
        self.assertEqual(cheap, 12400.0)
        self.assertEqual(dear, 14900.0)
        self.assertIsNone(parse_stated_number("Kestrel X2 pump"))
        self.assertIsNone(parse_stated_number("firmware v3"))
        self.assertIsNone(parse_stated_number("iPhone 15"))

    def test_parse_skips_quarter_years_and_iso_dates(self):
        self.assertEqual(parse_stated_number("Q3 2024 revenue was 1.2M"), 1_200_000.0)
        self.assertNotEqual(parse_stated_number("Q3 2024 revenue was 1.2M"), 3)
        self.assertNotEqual(parse_stated_number("Q3 2024 revenue was 1.2M"), 2024)
        self.assertIsNone(parse_stated_number("2024-03-05"))
        self.assertIsNone(parse_stated_number("signed on 2024-03-05"))

    def test_display_keeps_amount_on_blank_object_name(self):
        self.assertEqual(
            display_numeric_value("Halvorsen Dynamics founding", "was founded", 2011),
            "Halvorsen Dynamics founding 2011",
        )
        self.assertEqual(display_numeric_value("", "Priced per unit", 345), "Priced per unit 345")

    def test_retain_copies_year_onto_amount_not_event_name(self):
        rel = _Rel(description="Founded in 2011")
        original_name = rel.tip.name
        retain_numeric_on_relationship(rel)
        self.assertEqual(rel.amount, 2011.0)
        self.assertIn("2011", rel.description)
        self.assertEqual(rel.tip.name, original_name)
        self.assertEqual(rel.properties.get("amount"), 2011.0)

    def test_counter_values_get_distinct_relationship_ids(self):
        first = stable_relationship_id("org", "FOUNDED_IN", "founding", amount=2011)
        second = stable_relationship_id("org", "FOUNDED_IN", "founding", amount=2013)
        self.assertNotEqual(first, second)
        cheap = stable_relationship_id("sku", "PRICED_AT", "price", amount=12400)
        dear = stable_relationship_id("sku", "PRICED_AT", "price", amount=14900)
        self.assertNotEqual(cheap, dear)
        self.assertEqual(amount_identity_token(12400), "12400")
        self.assertTrue(amounts_differ(2011, 2013))
        self.assertTrue(amounts_differ(310, 345))
        self.assertFalse(amounts_differ(2011, 2011))

    def test_retain_keeps_two_price_facts_apart_without_renaming(self):
        first = _Rel(
            name="PRICED_AT",
            description="Kestrel X2 pump costs 12,400 EUR",
            tip=_Endpoint("unit price", "EVENT"),
        )
        second = _Rel(
            name="PRICED_AT",
            description="Kestrel X2 pump costs 14,900 EUR",
            tip=_Endpoint("unit price", "EVENT"),
        )
        first_name = first.tip.name
        second_name = second.tip.name
        retain_numeric_on_relationship(first)
        retain_numeric_on_relationship(second)
        self.assertEqual(first.amount, 12400.0)
        self.assertEqual(second.amount, 14900.0)
        self.assertTrue(amounts_differ(first.amount, second.amount))
        self.assertEqual(first.tip.name, first_name)
        self.assertEqual(second.tip.name, second_name)
        self.assertNotIn("12400", first.tip.name.replace(",", ""))
        self.assertNotIn("14900", second.tip.name.replace(",", ""))

    def test_retain_prefers_amount_then_source_span_over_node_names(self):
        named = _Rel(
            amount=14900,
            description="Kestrel X2 pump",
            tip=_Endpoint("X2 2024 costs", "EVENT"),
        )
        retain_numeric_on_relationship(named)
        self.assertEqual(named.amount, 14900.0)
        self.assertEqual(named.tip.name, "X2 2024 costs")

        spanned = _Rel(
            description="catalog row",
            source_span="Kestrel X2 pump costs 12,400 EUR",
            tip=_Endpoint("Q3 2024 listing", "EVENT"),
        )
        retain_numeric_on_relationship(spanned)
        self.assertEqual(spanned.amount, 12400.0)
        self.assertEqual(spanned.tip.name, "Q3 2024 listing")


if __name__ == "__main__":
    unittest.main()
