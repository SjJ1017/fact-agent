"""Turn a page fragment into a document a browser will actually render.

The templates are written for the Artifact host, which wraps them in a
doctype/html/head/body skeleton when it publishes. A file saved straight out of
one of them has none of that, so opening it from disk drops the browser into
quirks mode and the layout falls apart — the page looks broken everywhere
except on the host that adds the missing half.

`standalone` supplies that half. It is idempotent: a fragment that already
declares a doctype is returned untouched, so re-running a builder or piping a
finished page through this twice is safe.

`write_both` is what the builders call. The two consumers want opposite things
— a file opened from disk needs the skeleton, the Artifact host supplies its
own and asks you not to — so it writes the wrapped page where a person will
double-click it and the bare fragment next to the template for publishing.
Nesting one document inside the other does render, checked against a real
published page, but it relies on the parser discarding the stray tags, and that
is not a thing to depend on when the fix costs one extra file.
"""

from __future__ import annotations

HEAD_TAGS = ("<title", "<link", "<meta", "<style", "<base")


def standalone(fragment: str, lang: str = "zh-Hans") -> str:
    """Wrap a fragment in a full HTML document, head tags hoisted into <head>."""
    if fragment.lstrip()[:15].lower().startswith("<!doctype"):
        return fragment

    # Everything the template put at the top that belongs in <head> — the title,
    # the font links and the stylesheet — has to move there, or the browser
    # builds an empty head and the CSS lands mid-body.
    head: list[str] = []
    body = fragment
    while True:
        stripped = body.lstrip()
        if not stripped.lower().startswith(HEAD_TAGS):
            break
        tag = stripped[1:].split(">", 1)[0].split()[0].lower()
        closing = f"</{tag}>"
        if closing in stripped.lower():
            end = stripped.lower().index(closing) + len(closing)
        else:
            end = stripped.index(">") + 1
        head.append(stripped[:end])
        body = stripped[end:]

    indented = "\n".join("  " + line for line in head)
    return (
        f'<!doctype html>\n<html lang="{lang}">\n<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'{indented}\n</head>\n<body>\n{body.lstrip()}\n</body>\n</html>\n'
    )


def write_both(out, page: str) -> "tuple":
    """Write the standalone document to `out`, the fragment beside the template.

    Returns both paths. The fragment is named `<stem>.fragment.html` and lives
    in experiments/report/, so it is obvious which of the two to hand to a
    publisher and which to open.
    """
    from pathlib import Path

    out = Path(out)
    out.write_text(standalone(page))
    fragment = Path(__file__).resolve().parent / f"{out.stem}.fragment.html"
    fragment.write_text(page)
    return out, fragment
