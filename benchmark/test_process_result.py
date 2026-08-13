#!/usr/bin/env python3
"""Authentication regression tests for result ingestion."""

import os
import unittest
from unittest import mock

import process_result


class IngestAuthTests(unittest.TestCase):
    def test_bearer_header(self):
        with mock.patch.dict(os.environ, {"INGEST_BEARER_TOKEN": "test-token"}):
            headers = process_result.ingest_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-token")

    def test_missing_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "INGEST_BEARER_TOKEN"):
                process_result.ingest_headers()


if __name__ == "__main__":
    unittest.main()
