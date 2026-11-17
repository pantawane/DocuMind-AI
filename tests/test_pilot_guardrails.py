import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException
from langchain_core.documents import Document

from app.api import routes
from app.rag import vectorstore, workflow
from app.services.ingestion import chunk_documents


class IngestionCitationTests(unittest.TestCase):
    def test_chunks_keep_their_markdown_section(self):
        doc = Document(
            page_content="## Access Policy\n" + ("Employees must use MFA. " * 90),
            metadata={"source": "policy.md"},
        )

        chunks = chunk_documents([doc])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.metadata["section"] == "Access Policy" for chunk in chunks))

    def test_answer_citations_include_source_location_without_local_path(self):
        retrieved = Document(
            page_content="Reset passwords through the identity portal.",
            metadata={"source": str(Path("uploads") / "policy.pdf"), "page": 2, "section": "Account Access"},
        )
        fake_llm = Mock()
        fake_llm.invoke.return_value = SimpleNamespace(content="Use the identity portal [1].")

        with patch("app.rag.workflow.llm", return_value=fake_llm):
            result = workflow.generate_from_kb({
                "question": "How do I reset my password?",
                "kb_docs": [retrieved],
                "trace": [],
            })

        self.assertEqual(result["citations"][0]["id"], 1)
        self.assertEqual(result["citations"][0]["title"], "policy.pdf")
        self.assertEqual(result["citations"][0]["page"], 3)
        self.assertEqual(result["citations"][0]["section"], "Account Access")
        self.assertNotIn("uploads", result["citations"][0]["title"])
        prompt = fake_llm.invoke.call_args.args[0]
        self.assertIn("Source 1: policy.pdf page 3, section Account Access", prompt)


class UploadGuardrailTests(unittest.TestCase):
    def test_upload_is_disabled_without_an_admin_key(self):
        with patch.object(routes.settings, "admin_api_key", ""):
            with self.assertRaises(HTTPException) as raised:
                routes._validate_ingest_access("")
        self.assertEqual(raised.exception.status_code, 503)

    def test_upload_rejects_an_invalid_admin_key(self):
        with patch.object(routes.settings, "admin_api_key", "configured-secret"):
            with self.assertRaises(HTTPException) as raised:
                routes._validate_ingest_access("wrong-secret")
        self.assertEqual(raised.exception.status_code, 401)

    def test_upload_rejects_content_over_the_configured_limit(self):
        content = b"x" * (1024 * 1024 + 1)
        with patch.object(routes.settings, "max_upload_mb", 1):
            with self.assertRaises(HTTPException) as raised:
                routes._validate_upload_content(content)
        self.assertEqual(raised.exception.status_code, 413)

    def test_upload_rejects_empty_content(self):
        with self.assertRaises(HTTPException) as raised:
            routes._validate_upload_content(b"")
        self.assertEqual(raised.exception.status_code, 400)

    def test_upload_filename_is_normalized_and_type_checked(self):
        self.assertEqual(routes._normalize_upload_filename(r"C:\\private\\policy.md"), "policy.md")
        with self.assertRaises(HTTPException) as raised:
            routes._normalize_upload_filename("policy.exe")
        self.assertEqual(raised.exception.status_code, 400)

    def test_duplicate_upload_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as upload_dir:
            destination = Path(upload_dir, "policy.md")
            destination.write_text("old policy", encoding="utf-8")
            with self.assertRaises(HTTPException) as raised:
                routes._ensure_upload_destination_is_new(destination)
        self.assertEqual(raised.exception.status_code, 409)


class VectorIndexSafetyTests(unittest.TestCase):
    def test_dimension_mismatch_fails_without_deleting_existing_index(self):
        fake_pinecone = Mock()
        fake_pinecone.list_indexes.return_value = [{"name": "test-index"}]
        fake_pinecone.describe_index.return_value = SimpleNamespace(dimension=123)

        with patch.object(vectorstore.settings, "pinecone_api_key", "test-key"), \
             patch.object(vectorstore.settings, "pinecone_index_name", "test-index"), \
             patch.object(vectorstore.settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"), \
             patch("app.rag.vectorstore.Pinecone", return_value=fake_pinecone):
            with self.assertRaisesRegex(RuntimeError, "Create a new index and migrate data"):
                vectorstore.ensure_index()

        fake_pinecone.delete_index.assert_not_called()


if __name__ == "__main__":
    unittest.main()
