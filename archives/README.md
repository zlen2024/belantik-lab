# Legacy research archive

This archive contains local copies. The original experiments have not been deleted, and a complete backup on a separate drive has not yet been verified.

Contents under `legacy-projects/` are considered copied only when their SHA-256 manifest and batch report record a verified status. Investigate any other status before use. Metadata and the index do not invent unknown experimental outcomes. Git metadata, hidden files, videos and large data are in scope; junctions are recorded separately.

`archive_project.py` uses the frozen SHA-256 inventory, rechecks source and destination, and never modifies the original. It stops if the source changes, a file cannot be read, or a reparse point requires review. Matching local hashes are not a substitute for an off-device backup.

The [archive index](index.yaml) records each project’s status. Run `python archives/build_index.py` after copy reports change; `copy_in_progress` does not mean a completed archive. A `null` outcome means unknown, not success or failure.

A local restore test of `analog-lab-learn` compared 1,559 files and checked Git HEAD, branch, refs, worktree and status, including an uncommitted change. The test copy is in `restore-checks/`; it is not a separate legacy project. Recovery from another drive has not yet been tested. Some `.venv` directories and editable installs may contain old absolute paths, so matching hashes do not guarantee that an environment works after relocation.
