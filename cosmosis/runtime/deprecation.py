from . import logs

CONTACT = "Joe Zuntz"


def warn_deprecated(feature):
    """Log a (visible by default) warning that a feature is deprecated.

    feature should be a phrase like "The foo sampler" or "SMP process pools".
    """
    logs.warning(
        f"DEPRECATED: {feature} will probably be removed in a future version "
        f"of CosmoSIS. If you really want this feature kept, please get in "
        f"touch with {CONTACT}."
    )
