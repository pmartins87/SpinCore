from pathlib import Path

from spincore.deepcrusher_openppl import parse_openppl_source


def test_openppl_dependency_inventory(tmp_path: Path):
    source_path = tmp_path / "mini.txt"
    source_path.write_text(
        """##notes##\nmini\n"
        "##f$preflop##\nWhen f$helper && hand$A Return BetMax Force\n"
        "##f$flop##\nWhen user_plan && f$helper Return Call Force\n"
        "##f$turn##\nf$helper\n"
        "##f$river##\nWhen Others Fold Force\n"
        "##f$helper##\nf$native_external && user_plan\n"
        "##list_A##\nAA KK AKs\n"
        ,
        encoding="utf-8",
    )
    source = parse_openppl_source(source_path)
    inventory = source.inventory()
    assert inventory["missing_primary_roots"] == []
    assert inventory["function_sections"] == 5
    assert inventory["list_sections"] == 1
    assert "f$helper" in source.transitive_function_closure()
    assert source.external_f_symbols() == ("f$native_external",)
    assert source.referenced_hand_ranges() == ("hand$A",)
    assert source.referenced_user_variables() == ("user_plan",)
