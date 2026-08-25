"""Live-agent measurement pilot (Phase 1: collection + labelling).

See ``docs/live_pilot/strategy_definitions.md`` for the frozen instrument
this package implements against. Phase 1 uses only ``event_schema``,
``event_store``, ``tools``, ``agent_loop``, and ``labelling_cli`` — the
``ReplayResult`` schema and strategy-execution machinery exist for Phase 2
sign-off traceability but are not exercised by any Phase 1 code path.
"""
