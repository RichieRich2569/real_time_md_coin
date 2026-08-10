"""Exception hierarchy mirroring the MATLAB RealTimeCOIN error identifiers.

The MATLAB implementation raises errors with identifiers such as
``RealTimeCOIN:NoCues``; those identifiers are part of the behavioural
contract exercised by the test-suite, so every exception here carries its
identifier both as a class attribute and as a prefix of the message. That
makes ``pytest.raises(RealTimeCOINError, match="RealTimeCOIN:NoCues")``
work exactly as the MATLAB ``verifyError`` checks did.

Several classes inherit from a builtin exception in addition to
:class:`RealTimeCOINError` so that callers written against ordinary Python
conventions (``except ValueError``/``except TypeError``) still catch them.
"""

from __future__ import annotations

__all__ = [
    "RealTimeCOINError",
    "NoCuesError",
    "BiasNotInferredError",
    "ScalarModelOnlyError",
    "NameValuePairsError",
    "ModelFormatError",
]


class RealTimeCOINError(Exception):
    """Base class for every RealTimeCOIN error.

    Parameters
    ----------
    message : str, optional
        Human-readable description of the failure. The class identifier is
        prepended to it, so ``str(err)`` reads ``"<identifier>: <message>"``.

    Attributes
    ----------
    identifier : str
        The MATLAB error identifier this class stands in for.
    message : str
        The message without the identifier prefix.
    """

    identifier: str = "RealTimeCOIN:Error"

    def __init__(self, message: str = "") -> None:
        self.message = message
        super().__init__(f"{self.identifier}: {message}" if message else self.identifier)


class NoCuesError(RealTimeCOINError):
    """Raised when a cue-dependent quantity is requested but no cues exist."""

    identifier = "RealTimeCOIN:NoCues"


class BiasNotInferredError(RealTimeCOINError):
    """Raised when a bias read-out is requested with bias inference disabled."""

    identifier = "RealTimeCOIN:BiasNotInferred"


class ScalarModelOnlyError(RealTimeCOINError, ValueError):
    """Raised by read-outs that are only defined for ``state_dim == 1``."""

    identifier = "RealTimeCOIN:ScalarModelOnly"


class NameValuePairsError(RealTimeCOINError, TypeError):
    """Raised when constructor options are not given as name/value pairs."""

    identifier = "RealTimeCOIN:NameValuePairs"


class ModelFormatError(RealTimeCOINError, ValueError):
    """Raised when a persisted model file has an unexpected format."""

    identifier = "RealTimeCOIN:ModelFormat"
