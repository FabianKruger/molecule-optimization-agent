"""Deprecated console scripts that forward to seismo."""

import sys


def _notice(old: str, new: str) -> None:
    print(f"{old} is deprecated; use {new} instead", file=sys.stderr)


def molopt_main() -> None:
    _notice("molopt", "seismo")
    from seismo.main import main

    main()


def molopt_ui_main() -> None:
    _notice("molopt-ui", "seismo-ui")
    from seismo.ui.app import main

    main()
