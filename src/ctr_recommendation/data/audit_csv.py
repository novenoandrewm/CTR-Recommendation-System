"""Memory-conscious CSV integrity auditing using SQLite."""

# Import standard libraries
import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from ctr_recommendation.data.validate import (
    CandidateRecord,
    load_input_contract,
    validate_candidate,
    validate_header,
)


@dataclass(frozen=True)
class AuditResult:
    """Summary of a successfully validated CSV."""

    partition: str
    candidate_rows: int
    events: int
    clicks: int | None


def _context_hash(candidate: CandidateRecord) -> str:
    """Fingerprint the shared context of a recommendation event."""

    context = [
        candidate.user_id,
        candidate.timestamp,
        candidate.time_bucket,
        candidate.day_type,
        candidate.history_item_ids,
        candidate.history_timestamps,
    ]

    payload = json.dumps(context, separators=(",", ":"), ensure_ascii=False)

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _create_tables(connection: sqlite3.Connection) -> None:
    """Create disk-backed tables for global integrity checks."""

    connection.execute(
        """
        CREATE TABLE seen_rowids (
            candidate_rowid TEXT PRIMARY KEY
        ) WITHOUT ROWID
        """
    )

    connection.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            candidate_count INTEGER NOT NULL,
            click_count INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )


def _record_candidate(
    connection: sqlite3.Connection,
    candidate: CandidateRecord,
) -> None:
    """Record one candidate while checking global constraints."""

    try:
        connection.execute(
            "INSERT INTO seen_rowids VALUES (?)",
            (candidate.rowid,),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Duplicate rowid across dataset: {candidate.rowid}") from exc

    cursor = connection.execute(
        """
        INSERT INTO events (
            event_id,
            user_id,
            context_hash,
            candidate_count,
            click_count
        )
        VALUES (?, ?, ?, 1, ?)

        ON CONFLICT(event_id) DO UPDATE SET
            candidate_count = candidate_count + 1,
            click_count = click_count + excluded.click_count

        WHERE
            events.user_id = excluded.user_id
            AND events.context_hash = excluded.context_hash
        """,
        (
            candidate.event_id,
            candidate.user_id,
            _context_hash(candidate),
            int(candidate.clicked == 1),
        ),
    )

    if cursor.rowcount != 1:
        raise ValueError(f"Inconsistent context for event: {candidate.event_id}")


def audit_csv(
    csv_path: Path,
    contract_path: Path,
    *,
    partition: str,
) -> AuditResult:
    """Audit a complete candidate CSV without loading it into RAM."""

    contract = load_input_contract(contract_path)

    candidate_count = 0

    with tempfile.TemporaryDirectory(prefix="ctr_audit_") as temporary:
        database_path = Path(temporary) / "audit.sqlite"

        with closing(sqlite3.connect(database_path)) as connection:
            _create_tables(connection)

            with csv_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle, strict=True)

                if reader.fieldnames is None:
                    raise ValueError("CSV has no header")

                validate_header(
                    reader.fieldnames,
                    contract,
                    partition=partition,
                )

                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Malformed CSV record near line {reader.line_num}")

                    candidate = validate_candidate(
                        row,
                        contract,
                        partition=partition,
                    )

                    _record_candidate(connection, candidate)

                    candidate_count += 1

                    if candidate_count % 10000 == 0:
                        connection.commit()

            connection.commit()

            if candidate_count == 0:
                raise ValueError("CSV contains no candidate rows")

            expected = contract["events"]["expected_candidate_count"]

            maximum_clicks = contract["events"]["maximum_clicks_per_event"]

            invalid = connection.execute(
                """
                SELECT event_id, candidate_count, click_count
                FROM events

                WHERE candidate_count != ?
                   OR click_count > ?

                LIMIT 1
                """,
                (expected, maximum_clicks),
            ).fetchone()

            if invalid is not None:
                raise ValueError(
                    "Invalid event: "
                    f"event_id={invalid[0]}, "
                    f"candidates={invalid[1]}, "
                    f"clicks={invalid[2]}"
                )

            if partition == "evaluation":
                try:
                    connection.execute(
                        """
                        CREATE UNIQUE INDEX one_event_per_user
                        ON events(user_id)
                        """
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Evaluation contains multiple events for one user") from exc

            result = connection.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(click_count), 0)
                FROM events
                """
            ).fetchone()

            if result is None:
                raise RuntimeError("Failed to calculate event statistics")

            event_count, total_clicks = result

    return AuditResult(
        partition=partition,
        candidate_rows=candidate_count,
        events=event_count,
        clicks=total_clicks if partition == "train" else None,
    )


def main() -> None:
    """Run CSV validation from the command line."""

    parser = argparse.ArgumentParser(description="Audit a Huawei TechArena candidate CSV")

    parser.add_argument("--csv", required=True, type=Path)

    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("contracts/input_schema.json"),
    )

    parser.add_argument(
        "--partition",
        required=True,
        choices=("train", "evaluation"),
    )

    args = parser.parse_args()

    result = audit_csv(
        args.csv,
        args.contract,
        partition=args.partition,
    )

    print("CSV AUDIT: PASS")
    print(f"Partition: {result.partition}")
    print(f"Candidate rows: {result.candidate_rows}")
    print(f"Events: {result.events}")
    print(f"Clicks: {result.clicks}")


if __name__ == "__main__":
    main()
