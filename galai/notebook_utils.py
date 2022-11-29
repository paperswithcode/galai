from IPython.display import display, Markdown

__all__ = ["display_markdown"]


def display_markdown(text):
    # normalize LaTeX tags
    text = text.replace(r"\(", "$").replace(r"\)", "$").replace(r"\[", "$$").replace(r"\]", "$$")
    return display(Markdown(text))
