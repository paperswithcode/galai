from IPython.display import HTML
import markdown as md
import bleach
from bleach.css_sanitizer import CSSSanitizer


__all__ = ["display_markdown", "display_latex"]

ALLOWED_TAGS = [
    "a",
    "abbr",
    "acronym",
    "b",
    "blockquote",
    "br",
    "code",
    "div",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "i",
    "li",
    "ol",
    "strong",
    "ul",
    "span",
    "table",
    "thead",
    "tbody",
    "tr",
    "td",
    "th",
    "p",
    "pre",
]

ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "abbr": ["title"],
    "acronym": ["title"],
    "div": ["class"],
    "span": ["style", "class"],
    "td": ["align", "valign"],
    "th": ["align", "valign"],
}

ALLOWED_CSS_PROPERTIES = [
    "width", "margin", "margin-left", "margin-right",
    "margin-bottom", "margin-top", "height", "color", "font-weight"
]


def clean_html(value, tags=None, attributes=None, css_sanitizer=None):
    if tags is None:
        tags = ALLOWED_TAGS
    if attributes is None:
        attributes = ALLOWED_ATTRIBUTES
    if css_sanitizer is None:
        css_sanitizer = CSSSanitizer(allowed_css_properties=ALLOWED_CSS_PROPERTIES)
    elif isinstance(css_sanitizer, list):
        css_sanitizer = CSSSanitizer(allowed_css_properties=css_sanitizer)

    cleaned = bleach.clean(
        value,
        tags=tags,
        attributes=attributes,
        css_sanitizer=css_sanitizer,
    )

    return cleaned


def _markdown2html_unsafe(value):
    """Converts markdown to unsanitized HTML."""
    out = md.markdown(
        value,
        extensions=[
            "markdown.extensions.tables", "fenced_code", "codehilite"
        ],
    )
    return out


def markdown2html(value):
    return clean_html(_markdown2html_unsafe(value))


def display_markdown(text):
    # normalize LaTeX tags
    text = text.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")
    # convert to markdown and sanitize
    text = markdown2html(text)
    # use IPython.display.HTML instead of IPython.display.Markdown so that the output is
    # rendered properly on notebook load without cells reevaluations
    return HTML(text)


def display_latex(text):
    # normalize LaTeX tags
    text = text.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")
    # the text is going to be parsed as
    text = clean_html(text, tags=[], attributes=[], css_sanitizer=[])
    # use IPython.display.HTML instead of IPython.display.Latex so that the output is
    # rendered properly on notebook load without cells reevaluations
    return HTML(text)
