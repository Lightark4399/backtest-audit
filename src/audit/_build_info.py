"""Optional metadata stamped by a release build.

Source and editable installs deliberately report an unknown build commit. A
release pipeline may replace this value when the source revision is known.
"""

BUILD_COMMIT: str | None = None
