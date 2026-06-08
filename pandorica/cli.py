#######################################################################
#  Serial Stitcher - An Automatic tool for tomograms stitching        #
#                                                                     #
#  https://github.com/RRobert92                                       #
#                                                                     #
#  Robert Kiewisz                                                     #
#  PolyForm Noncommercial License 1.0.0 - see LICENSE                 #
#######################################################################
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Robert Kiewisz

"""
Pandorica's command-line interface.

A ``pandorica`` console script with subcommands; the only one wired today is
``pandorica stitch``. Flags for ``stitch`` are derived at import time from
:func:`pandorica.stitch.cli.run_stitch` (same introspection pattern as the
``tardis_stitch`` wrapper in ``tardis_em``) — adding/renaming a parameter in
``run_stitch`` automatically updates the CLI without code changes here.

Rich rendering lives **only in this module**. :func:`pandorica.stitch.cli
.run_stitch` stays library-pure and continues to default ``log=print``, so
external callers (e.g. ``tardis_stitch``) keep their existing output style
without any rich frames colliding with theirs. The rich panel + styled log
callback are wired up by this CLI alone.
"""

from __future__ import annotations

import inspect
import re
import typing
import warnings
from datetime import datetime

import click
from docstring_parser import parse as _parse_docstring
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from pandorica._version import version as _VERSION
except Exception:  # noqa: BLE001
    _VERSION = "?"

warnings.simplefilter("ignore", UserWarning)


# ---------------------------------------------------------------------------
# Stream-based pretty log
# ---------------------------------------------------------------------------


_SECTION_RE = re.compile(r"^---\s+(.*?)\s+---\s*$")
_WARNING_LINE = "!"


class _PrettyLog:
    """Stream the report lines emitted by ``run_stitch`` and render them with rich.

    ``run_stitch`` writes a structured text report one line at a time via its
    ``log`` callable. We classify each line on the fly:

    * ``--- foo ---`` headers become :class:`rich.rule.Rule` dividers.
    * ``! ...`` lines (the image-only-mode warning block) collect into one
      yellow warning panel.
    * ``ok`` / ``FAIL`` / ``accepted`` tokens get green / red highlighting.
    * Any line whose first non-space character is ``#`` is **dropped** from
      the terminal — that's the banner/citation block, which the CLI already
      replaced with its own rich panel. The plain-text version still lands
      in ``stitch_log.txt`` because ``run_stitch`` writes the full report
      list to disk regardless of what the log callback does.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._warn_buf: list[str] = []

    # ----- internal helpers ------------------------------------------------

    def _flush_warn(self) -> None:
        if not self._warn_buf:
            return
        text = "\n".join(self._warn_buf).strip()
        self.console.print(
            Panel(text, title="warning", border_style="yellow", expand=False)
        )
        self._warn_buf = []

    def _style_status_tokens(self, line: str) -> Text:
        """Style ``ok`` / ``FAIL`` / ``True``/``False`` tokens inline."""
        styled = Text(line)
        for token, style in (
            ("FAIL", "bold red"),
            ("ok", "green"),
            ("accepted", "green"),
            ("rejected", "bold red"),
        ):
            for m in re.finditer(rf"\b{re.escape(token)}\b", line):
                styled.stylize(style, m.start(), m.end())
        return styled

    # ----- callback --------------------------------------------------------

    def __call__(self, line: str) -> None:
        # Banner / citation block from run_stitch: every line in it starts
        # with ``#`` (border, title, BibTeX entry, upstream-citation list).
        # Drop them all — the CLI showed its own rich panel before
        # run_stitch was invoked; the full text remains in stitch_log.txt.
        if line.lstrip().startswith("#"):
            return

        # ``!`` warning block from the image-only branch: buffer until the
        # block ends (next non-``!`` line), then emit as one yellow panel.
        if line.startswith(_WARNING_LINE):
            stripped = line.lstrip("!").lstrip()
            if stripped:
                self._warn_buf.append(stripped)
            return
        if self._warn_buf:
            self._flush_warn()

        # Section dividers (``--- foo ---``) → rich.rule.
        m = _SECTION_RE.match(line)
        if m:
            self.console.print(Rule(m.group(1), style="cyan"))
            return

        if line == "":
            self.console.print()
            return

        self.console.print(self._style_status_tokens(line))

    # ----- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Flush any pending warning panel."""
        self._flush_warn()


# ---------------------------------------------------------------------------
# Startup / summary panels
# ---------------------------------------------------------------------------


def _startup_panel(console: Console, input_dir: str, output_dir: str | None) -> None:
    out = output_dir or f"{input_dir.rstrip('/')}/stitched_output"
    body = Text()
    body.append("pandorica ", style="bold")
    body.append(f"v{_VERSION}", style="cyan")
    body.append("  ·  serial-section stitcher\n", style="dim")
    body.append(f"started   : {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    body.append(f"input_dir : {input_dir}\n")
    body.append(f"output_dir: {out}")
    console.print(Panel(body, title="PANDORICA", border_style="cyan", expand=False))


def _summary_panel(console: Console, written: dict) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    for k in ("volume", "graph", "log"):
        if k in written:
            table.add_row(k, str(written[k]))
    if "seconds" in written:
        table.add_row("elapsed", f"{written['seconds']:.1f} s")
    console.print(
        Panel(table, title="done", border_style="green", expand=False)
    )


def _error_panel(console: Console, message: str, title: str = "error") -> None:
    console.print(Panel(message, title=title, border_style="red", expand=False))


# ---------------------------------------------------------------------------
# stitch subcommand — auto-derived from run_stitch's signature
# ---------------------------------------------------------------------------


def _unwrap_optional(ann):
    """``Optional[T]`` → ``T``; pass-through otherwise."""
    if typing.get_origin(ann) is typing.Union:
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return ann


def _is_exposable(p: inspect.Parameter) -> bool:
    if p.annotation is inspect.Parameter.empty:
        return False
    if callable(p.default) and not isinstance(p.default, type):
        return False
    return True


def _help_map(fn) -> dict[str, str]:
    if not fn.__doc__:
        return {}
    parsed = _parse_docstring(inspect.getdoc(fn) or "")
    return {
        p.arg_name: " ".join((p.description or "").split())
        for p in parsed.params
        if (p.description or "").strip()
    }


def _build_stitch_options() -> list[click.Option]:
    from pandorica.stitch.cli import run_stitch

    sig = inspect.signature(run_stitch)
    helps = _help_map(run_stitch)
    options: list[click.Option] = []
    for name, p in sig.parameters.items():
        if not _is_exposable(p):
            continue
        ann = _unwrap_optional(p.annotation)
        has_default = p.default is not inspect.Parameter.empty
        default = p.default if has_default else None
        required = not has_default
        flag = name.replace("_", "-")
        help_text = helps.get(name, "")
        if ann is bool:
            options.append(
                click.Option(
                    [f"--{flag}/--no-{flag}"],
                    default=default,
                    show_default=True,
                    help=help_text,
                )
            )
        else:
            # click only enforces ``required`` when no ``default`` is passed at all;
            # an explicit ``default=None`` silently satisfies it (the callback would
            # then run with None). So omit ``default`` for required params.
            opt_kwargs = dict(
                type=ann, required=required, show_default=True, help=help_text,
            )
            if has_default:
                opt_kwargs["default"] = default
            options.append(click.Option([f"--{flag}"], **opt_kwargs))
    return options


def _stitch_short_help() -> str:
    from pandorica.stitch.cli import run_stitch

    d = _parse_docstring(inspect.getdoc(run_stitch) or "")
    return (
        d.short_description
        or "Stitch serial-section tomograms into one volume + merged graph."
    )


def _stitch_callback(**kwargs) -> None:
    from pandorica.stitch.cli import run_stitch

    console = Console(stderr=False, highlight=False)
    _startup_panel(console, kwargs.get("input_dir", "?"), kwargs.get("output_dir"))
    pretty = _PrettyLog(console)
    try:
        written = run_stitch(log=pretty, **kwargs)
    except (OSError, ValueError) as e:
        # OSError covers FileNotFoundError (input dir missing) and
        # PermissionError / read-only filesystem on output makedirs;
        # ValueError covers "nothing to stitch" and similar validation.
        pretty.close()
        _error_panel(console, str(e), title="cannot stitch")
        raise SystemExit(1)
    except KeyboardInterrupt:
        pretty.close()
        _error_panel(console, "interrupted by user", title="aborted")
        raise SystemExit(130)
    pretty.close()
    _summary_panel(console, written)


def _build_stitch_command() -> click.Command:
    return click.Command(
        "stitch",
        params=_build_stitch_options(),
        callback=_stitch_callback,
        help=_stitch_short_help(),
        no_args_is_help=True,  # bare ``pandorica stitch`` prints help, not an error
    )


# ---------------------------------------------------------------------------
# Top-level ``pandorica`` group
# ---------------------------------------------------------------------------


@click.group(
    name="pandorica",
    help="Pandorica — serial-section tomogram stitcher and related tools.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version=_VERSION, prog_name="pandorica")
def main() -> None:
    """Group root; subcommands do the work."""


main.add_command(_build_stitch_command())


if __name__ == "__main__":
    main()
