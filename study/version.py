"""Protocol version for the blind-design elicitation study.

Separate from ``study.storage.SCHEMA_VERSION`` (the on-disk record
format) and ``study.storage.DATASET_VERSION`` (the released artifact) --
this one changes only when ``study/protocol.md``'s actual protocol
(what's shown, what's asked, the locking rule) changes. Bumping this
without bumping the others is expected and normal (e.g. a wording fix to
a description changes the protocol but not the storage schema).
"""

PROTOCOL_VERSION = "1.0.0"
