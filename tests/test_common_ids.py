import re

from src.propensity.common.ids import generate_id, seed_from_id, hex_seed_from_id


def test_generate_id_matches_expected_format():
    id_string = generate_id()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-[0-9a-f]{4}", id_string)


def test_generate_id_uses_given_hex_seed():
    id_string = generate_id(hex_seed="00ff")
    assert id_string.endswith("-00ff")


def test_hex_seed_from_id_round_trips():
    id_string = generate_id(hex_seed="1a2b")
    assert hex_seed_from_id(id_string) == "1a2b"


def test_seed_from_id_matches_hex_value():
    id_string = generate_id(hex_seed="00ff")
    assert seed_from_id(id_string) == 255


def test_generate_id_is_reproducible_given_same_seed():
    assert generate_id(hex_seed="abcd") == generate_id(hex_seed="abcd")
