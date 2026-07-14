# Annotation output directory

Default exports are written here with a session timestamp and inclusive episode
positions in the filename:

```text
<input_stem>_annotated_YYYYMMDDTHHMMSS_microseconds_episodes_201-300.jsonl
```

Repeated saves of the same range within a session atomically update that range
file. Different assignment ranges therefore remain separate. On the next
startup, compatible range files newer than the latest full snapshot are replayed
in chronological order to restore saved progress.

Generated JSONL files are ignored by Git; only this README is tracked.
