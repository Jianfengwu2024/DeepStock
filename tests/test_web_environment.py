from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bnhmm.web.app import _load_environment_file


class WebEnvironmentTests(unittest.TestCase):
    def test_dotenv_loading_supports_comments_quotes_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory, ".env")
            env_file.write_text(
                "# deployment values\n"
                "BNHMM_TEST_QUOTED='value with spaces'\n"
                "export BNHMM_TEST_EXPORTED=loaded # comment\n"
                "BNHMM_TEST_EXISTING=from-file\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"BNHMM_TEST_EXISTING": "from-process"},
                clear=True,
            ):
                _load_environment_file(env_file)

                self.assertEqual(
                    os.environ["BNHMM_TEST_QUOTED"], "value with spaces"
                )
                self.assertEqual(os.environ["BNHMM_TEST_EXPORTED"], "loaded")
                self.assertEqual(
                    os.environ["BNHMM_TEST_EXISTING"], "from-process"
                )


if __name__ == "__main__":
    unittest.main()
