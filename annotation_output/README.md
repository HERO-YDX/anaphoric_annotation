# Annotation output directory

Default exports are written here with a timestamped name:

```text
<input_stem>_annotated_YYYYMMDDTHHMMSS_microseconds.jsonl
```

Each server session gets a new filename. Repeated **Save All to File** actions
within that session atomically update the same file. When the application is
started again, it restores annotations from the newest compatible timestamped
file and writes subsequent saves to a new version.

Generated JSONL files are ignored by Git; only this README is tracked.
