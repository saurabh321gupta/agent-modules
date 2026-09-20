"""Which controls may be shown to a model, and which are worth showing. Layer 1.

This is the only place that decides either question, so the classifier's view and the reader's own
notions can never disagree about it.
"""

from __future__ import annotations

from .models import PageElement


def is_capturable(element: PageElement) -> bool:
    """Whether this control is part of the page at all.

    Anchors are navigation, not application controls, and are never sent.

    A file input counts even when it is not visible. A site normally hides the real
    `<input type="file">` and styles a button in front of it, so visibility says nothing about
    whether the control matters - and `set_input_files` works on a hidden input perfectly well.
    Dropping it makes the field invisible to every later stage, which is how a required resume
    upload silently never happens and the form refuses to advance.
    """
    if element.role == "link":
        return False
    if element.input_type == "file":
        return True
    return element.visible


def is_actionable(element: PageElement) -> bool:
    """Whether a model could have anything to *do* with this control.

    A file input that already holds a file is the case this exists for, and it is not hypothetical:
    an application wizard keeps its resume widget mounted in the page shell, so the same hidden input
    appears on every step, already attached after the first upload. Showing it invites the model to
    upload the resume again on page four - and because re-uploading the same file changes nothing,
    the identical plan repeats until the run is declared stuck. There is nothing to do with an
    attached file.
    """
    if element.input_type == "file" and element.file_attached:
        return False
    return is_capturable(element)
