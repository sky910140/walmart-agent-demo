from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finagent.agent import FinancialAgent
from finagent.ingest import build_filing_index, html_to_text
from finagent.market import market_snapshot
from finagent.memory import PreferenceStore
from finagent.models import ModelGateway
from finagent.retrieval import LocalRetriever, chunk_document
from finagent.sec import download_sec_10k, validate_sec_user_agent
from finagent.websearch import WebResult, _result_url


class FinancialAgentTests(unittest.TestCase):
    def test_chunks_keep_verifiable_source_metadata(self) -> None:
        chunks = chunk_document(
            document_id="acme-2025-10k",
            title="Acme 2025 Form 10-K",
            text="Liquidity risk increased because debt maturities concentrate in 2027. " * 8,
            source_url="https://www.sec.gov/Archives/example.htm",
            published_at="2025-02-20",
            source_type="sec_10k",
            chunk_size=120,
            overlap=20,
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_url, "https://www.sec.gov/Archives/example.htm")
        self.assertEqual(chunks[0].source_type, "sec_10k")
        self.assertEqual(chunks[0].chunk_id, "acme-2025-10k:0001")

    def test_retrieval_ranks_relevant_evidence_and_returns_citation(self) -> None:
        chunks = chunk_document(
            "acme-2025-10k",
            "Acme 2025 Form 10-K",
            "The company faces liquidity risk from a revolving credit facility and debt maturities.",
            "https://www.sec.gov/Archives/example.htm",
            "2025-02-20",
            "sec_10k",
        ) + chunk_document(
            "acme-2025-10k",
            "Acme 2025 Form 10-K",
            "The company sells consumer devices through retail partners.",
            "https://www.sec.gov/Archives/example.htm",
            "2025-02-20",
            "sec_10k",
        )
        results = LocalRetriever(chunks).search("summarize liquidity and debt risk", limit=1)

        self.assertEqual(len(results), 1)
        self.assertIn("liquidity", results[0].evidence.text.lower())
        self.assertEqual(results[0].citation.label, "[S1]")

    def test_market_snapshot_calculates_period_change_and_keeps_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "csi300.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["date", "close", "volume"])
                writer.writeheader()
                writer.writerows([
                    {"date": "2024-01-02", "close": "3300", "volume": "1200000"},
                    {"date": "2024-01-03", "close": "3400", "volume": "1500000"},
                ])
            Path(f"{csv_path}.meta.json").write_text(json.dumps({"source_url": "https://query1.finance.yahoo.com/example"}), encoding="utf-8")

            snapshot = market_snapshot(csv_path, start="2024-01-02", end="2024-01-03")

        self.assertEqual(snapshot.start_close, 3300.0)
        self.assertEqual(snapshot.end_close, 3400.0)
        self.assertAlmostEqual(snapshot.change_percent, 3.03, places=2)
        self.assertEqual(snapshot.source_url, "https://query1.finance.yahoo.com/example")

    def test_preferences_are_persisted_and_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PreferenceStore(Path(directory) / "memory.json")
            store.record("alice", "I care most about liquidity risk and debt maturity.")
            store.record("alice", "Please focus on cash flow too.")

            preferences = store.get("alice")

        self.assertEqual(preferences, ["cash flow", "debt maturity", "liquidity risk"])

    def test_model_gateway_is_explicit_when_keys_are_missing(self) -> None:
        gateway = ModelGateway(doubao_api_key=None, deepseek_api_key=None)
        result = gateway.complete("doubao", "system", "user")

        self.assertEqual(result.provider, "offline")
        self.assertFalse(result.used_remote_model)
        self.assertNotIn("api_key", result.text.lower())

    def test_index_and_agent_produce_an_offline_cited_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            (docs / "acme.html").write_text(
                "<html><body><h1>Risk Factors</h1><p>" + ("Acme faces liquidity risk because debt maturities are concentrated in 2027. " * 8) + "</p></body></html>",
                encoding="utf-8",
            )
            (docs / "manifest.jsonl").write_text(json.dumps({
                "document_id": "acme-2025-10k",
                "company": "Acme Inc.",
                "form": "10-K",
                "report_date": "2025-12-31",
                "filing_date": "2026-02-20",
                "accession_number": "000000-26-000001",
                "source_url": "https://www.sec.gov/Archives/acme.htm",
                "local_path": "acme.html",
            }) + "\n", encoding="utf-8")
            index = root / "index.json"
            documents, chunks = build_filing_index(docs, index, chunk_size=120)
            response = FinancialAgent(
                index_path=index,
                memory_path=root / "memory.json",
                market_path=root / "missing.csv",
                models=ModelGateway(doubao_api_key=None, deepseek_api_key=None),
            ).ask("Please focus on liquidity risk and debt maturity.", user_id="alice")

        self.assertEqual(documents, 1)
        self.assertGreater(chunks, 0)
        self.assertIn("[S1]", response.answer)
        self.assertEqual(response.preferences, ["debt maturity", "liquidity risk"])
        self.assertEqual(response.citations[0].source_type, "sec_10k")

    def test_web_results_are_explicitly_labelled_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "finagent.agent.search_public_web",
            return_value=[WebResult("SEC filing page", "https://www.sec.gov/example", "Public result snippet")],
        ):
            root = Path(directory)
            response = FinancialAgent(
                index_path=root / "absent-index.json",
                memory_path=root / "memory.json",
                market_path=root / "missing.csv",
            ).ask("Find this company filing", include_web=True)

        self.assertEqual(response.evidence_count, 1)
        self.assertEqual(response.citations[0].source_type, "web_search")
        self.assertIn("Public result snippet", response.answer)

    def test_sec_downloader_writes_source_manifest(self) -> None:
        submissions = {"filings": {"recent": {
            "form": ["10-K"],
            "accessionNumber": ["000001-26-000001"],
            "primaryDocument": ["annual.htm"],
            "filingDate": ["2026-02-20"],
            "reportDate": ["2025-12-31"],
        }}}
        with tempfile.TemporaryDirectory() as directory, patch("finagent.sec._get_json", return_value=submissions), patch(
            "finagent.sec._get_bytes", return_value=b"<html><body>Annual report</body></html>"
        ):
            records = download_sec_10k(
                Path(directory),
                user_agent="FinancialAgent test@example.com",
                companies=(("ACME", "Acme Inc.", 1),),
            )
            manifest = Path(directory) / "manifest.jsonl"

            row = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(len(records), 1)
        self.assertEqual(row["source_type"], "sec_10k")
        self.assertIn("www.sec.gov/Archives", row["source_url"])

    def test_sec_user_agent_requires_contact_information(self) -> None:
        with self.assertRaises(ValueError):
            validate_sec_user_agent("FinancialAgent")

    def test_html_text_normalizes_typographic_punctuation_for_cli_output(self) -> None:
        self.assertEqual(html_to_text("<p>Company&#8217;s debt - 2025</p>"), "Company's debt - 2025")

    def test_web_search_unwraps_duckduckgo_redirects(self) -> None:
        self.assertEqual(
            _result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.sec.gov%2Fexample&amp;rut=abc"),
            "https://www.sec.gov/example",
        )


if __name__ == "__main__":
    unittest.main()
