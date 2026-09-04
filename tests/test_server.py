"""Unit test specifications for the TPU LLO Viewer remote web server.

This test suite serves as the executable contract and behavioral specification
for `server.py`, validating argument parsing, filesystem security guards, run
and file discovery, multi-tier remote source resolution, and HTTP REST API
endpoints served by `LLOServerHandler`.
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import runpy
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.parse
import urllib.request

from server import (
    LLOServerHandler,
    create_server,
    discover_runs,
    is_safe_path,
    list_run_files,
    main,
    parse_args,
    resolve_source_file,
)
try:
    from server import clear_source_cache
except ImportError:
    clear_source_cache = lambda: None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DATA_DIR = REPO_ROOT / "example_data" / "data"
EXAMPLE_SOURCE_FILE = REPO_ROOT / "example_data" / "bench_pcp_q_compute_micro.py"


class TestParseArgs(unittest.TestCase):
    """Behavioral specifications for CLI argument parsing in `parse_args`."""

    def test_should_return_default_arguments_when_no_flags_provided(self) -> None:
        """Verifies default values for CLI flags when no CLI options are specified.

        Asserts that:
            - `--dir` defaults to `.`
            - `--port` defaults to 8080 (int)
            - `--host` defaults to `0.0.0.0`
            - `--workspace-root` defaults to `.`
            - `--max-depth` defaults to 6 (int)
        """
        # Arrange
        cli_args: list[str] = []

        # Act
        args = parse_args(cli_args)

        # Assert
        self.assertEqual(args.dir, ".")
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.workspace_root, ".")
        self.assertEqual(args.max_depth, 6)

    def test_should_parse_custom_directory_and_port_when_specified(self) -> None:
        """Verifies custom values for directory path and network port.

        Asserts that `--dir` and `--port` override defaults correctly.
        """
        # Arrange
        cli_args = ["--dir", "/custom/dump/path", "--port", "9090"]

        # Act
        args = parse_args(cli_args)

        # Assert
        self.assertEqual(args.dir, "/custom/dump/path")
        self.assertEqual(args.port, 9090)

    def test_should_parse_custom_host_and_workspace_root_when_specified(self) -> None:
        """Verifies custom values for binding host and workspace root.

        Asserts that `--host` and `--workspace-root` flags are honored.
        """
        # Arrange
        cli_args = ["--host", "127.0.0.1", "--workspace-root", "/home/repo/workspace"]

        # Act
        args = parse_args(cli_args)

        # Assert
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.workspace_root, "/home/repo/workspace")

    def test_should_parse_custom_max_depth_when_specified(self) -> None:
        """Verifies custom value for max directory search depth.

        Asserts that `--max-depth` is converted to an integer properly.
        """
        # Arrange
        cli_args = ["--max-depth", "3"]

        # Act
        args = parse_args(cli_args)

        # Assert
        self.assertEqual(args.max_depth, 3)


class TestIsSafePath(unittest.TestCase):
    """Behavioral specifications for path traversal security guard `is_safe_path`."""

    def setUp(self) -> None:
        """Creates a temporary directory hierarchy with nested files and symlinks."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name) / "base"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.valid_file = self.base_dir / "valid.txt"
        self.valid_file.write_text("valid content", encoding="utf-8")

        self.valid_subdir = self.base_dir / "sub"
        self.valid_subdir.mkdir(parents=True, exist_ok=True)
        self.nested_file = self.valid_subdir / "nested.txt"
        self.nested_file.write_text("nested content", encoding="utf-8")

        self.outside_dir = Path(self.temp_dir.name) / "outside"
        self.outside_dir.mkdir(parents=True, exist_ok=True)
        self.outside_file = self.outside_dir / "secret.txt"
        self.outside_file.write_text("secret content", encoding="utf-8")

    def tearDown(self) -> None:
        """Cleans up the temporary directory hierarchy."""
        self.temp_dir.cleanup()

    def test_should_allow_path_when_path_is_inside_base_directory(self) -> None:
        """Verifies that files and subdirectories located within the base directory are allowed."""
        # Arrange
        target_path = self.nested_file

        # Act
        is_safe = is_safe_path(self.base_dir, target_path)

        # Assert
        self.assertTrue(is_safe)

    def test_should_allow_base_directory_itself(self) -> None:
        """Verifies that the base directory path itself is evaluated as safe."""
        # Arrange & Act
        is_safe = is_safe_path(self.base_dir, self.base_dir)

        # Assert
        self.assertTrue(is_safe)

    def test_should_reject_path_when_path_traverses_outside_base_directory(self) -> None:
        """Verifies that relative path traversal sequences (e.g. `../../etc/passwd`) are rejected."""
        # Arrange
        traversal_path = os.path.join(str(self.base_dir), "..", "outside", "secret.txt")

        # Act
        is_safe = is_safe_path(self.base_dir, traversal_path)

        # Assert
        self.assertFalse(is_safe)

    def test_should_reject_path_when_absolute_path_is_outside_base_directory(self) -> None:
        """Verifies that an absolute path residing outside the base directory is rejected."""
        # Arrange
        target_path = self.outside_file

        # Act
        is_safe = is_safe_path(self.base_dir, target_path)

        # Assert
        self.assertFalse(is_safe)

    def test_should_allow_symlink_when_target_resolves_inside_base_directory(self) -> None:
        """Verifies that symlinks resolving to targets inside the base directory are allowed."""
        # Arrange
        safe_symlink = self.base_dir / "symlink_valid.txt"
        safe_symlink.symlink_to(self.valid_file)

        # Act
        is_safe = is_safe_path(self.base_dir, safe_symlink)

        # Assert
        self.assertTrue(is_safe)

    def test_should_reject_symlink_when_target_resolves_outside_base_directory(self) -> None:
        """Verifies that symlinks pointing outside the base directory are detected and rejected."""
        # Arrange
        escape_symlink = self.base_dir / "symlink_escape.txt"
        escape_symlink.symlink_to(self.outside_file)

        # Act
        is_safe = is_safe_path(self.base_dir, escape_symlink)

        # Assert
        self.assertFalse(is_safe)

    def test_should_return_false_when_commonpath_raises_value_error(self) -> None:
        """Verifies False is returned when commonpath raises ValueError (e.g. cross-drive)."""
        with patch("os.path.commonpath", side_effect=ValueError("different drives")):
            self.assertFalse(is_safe_path("/base/dir", "/base/dir/file.txt"))

    def test_should_reject_path_when_target_shares_directory_prefix_collision(self) -> None:
        """Verifies prefix collision protection (e.g. /base vs /base_evil)."""
        prefix_collision_dir = Path(str(self.base_dir) + "_evil")
        prefix_collision_dir.mkdir(parents=True, exist_ok=True)
        self.assertFalse(is_safe_path(self.base_dir, prefix_collision_dir))


class TestDiscoverRuns(unittest.TestCase):
    """Behavioral specifications for LLO run discovery in `discover_runs`."""

    def test_should_discover_runs_when_bundles_exist_in_example_data(self) -> None:
        """Verifies that all runs containing `*-final_bundles.txt` in example_data are discovered."""
        # Arrange
        search_dir = EXAMPLE_DATA_DIR

        # Act
        runs = discover_runs(search_dir)

        # Assert
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 7)

    def test_should_return_required_fields_for_discovered_runs(self) -> None:
        """Verifies that every discovered run dictionary conforms to the required JSON schema."""
        # Arrange
        search_dir = EXAMPLE_DATA_DIR

        # Act
        runs = discover_runs(search_dir)

        # Assert
        self.assertTrue(len(runs) > 0)
        for run in runs:
            self.assertIn("id", run)
            self.assertIn("name", run)
            self.assertIn("bundleCount", run)
            self.assertIn("mtime", run)
            self.assertIsInstance(run["id"], str)
            self.assertIsInstance(run["name"], str)
            self.assertIsInstance(run["bundleCount"], int)
            self.assertGreaterEqual(run["bundleCount"], 1)
            self.assertIsInstance(run["mtime"], (int, float))

    def test_should_return_empty_list_when_no_bundle_files_found(self) -> None:
        """Verifies that scanning a directory with no `*-final_bundles.txt` returns an empty list."""
        # Arrange
        with tempfile.TemporaryDirectory() as empty_temp_dir:
            # Act
            runs = discover_runs(empty_temp_dir)

            # Assert
            self.assertEqual(runs, [])

    def test_should_respect_max_depth_limit(self) -> None:
        """Verifies that `max_depth` limits directory traversal."""
        # Arrange
        # Bundles in EXAMPLE_DATA_DIR are located at depth >= 4
        search_dir = EXAMPLE_DATA_DIR

        # Act
        shallow_runs = discover_runs(search_dir, max_depth=1)

        # Assert
        self.assertEqual(shallow_runs, [])

    def test_should_return_empty_list_when_search_dir_does_not_exist(self) -> None:
        """Verifies discover_runs returns empty list when directory does not exist."""
        nonexistent = REPO_ROOT / "nonexistent_dir_abc_xyz"
        self.assertEqual(discover_runs(nonexistent), [])

    def test_should_skip_files_beyond_max_depth_and_handle_stat_error(self) -> None:
        """Verifies discover_runs handles stat errors and falls back to root mtime."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run1"
            run_dir.mkdir()
            (run_dir / "op-final_bundles.txt").write_text("dummy")

            orig_getmtime = os.path.getmtime

            def mock_getmtime(p: str) -> float:
                if "final_bundles.txt" in p:
                    raise OSError("bundle stat failed")
                return orig_getmtime(p)

            with patch("os.path.getmtime", side_effect=mock_getmtime):
                runs = discover_runs(tmp_path)
                self.assertEqual(len(runs), 1)
                self.assertGreater(runs[0]["mtime"], 0.0)

    def test_should_fallback_to_zero_mtime_when_all_mtimes_fail(self) -> None:
        """Verifies discover_runs sets mtime to 0.0 when root stat also fails."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run2"
            run_dir.mkdir()
            (run_dir / "op-final_bundles.txt").write_text("dummy")

            with patch("os.path.getmtime", side_effect=OSError("stat error")):
                runs = discover_runs(tmp_path)
                self.assertEqual(len(runs), 1)
                self.assertEqual(runs[0]["mtime"], 0.0)

    def test_should_return_empty_when_max_depth_is_negative(self) -> None:
        """Verifies discover_runs returns empty list and skips root when max_depth is negative."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "test-final_bundles.txt").write_text("dummy")
            self.assertEqual(discover_runs(tmp_path, max_depth=-1), [])

    def test_should_prune_symlinks_pointing_outside_search_dir(self) -> None:
        """Verifies discover_runs prunes symlinks pointing outside search_dir."""
        with tempfile.TemporaryDirectory() as temp_dir:
            search_dir = Path(temp_dir) / "search"
            search_dir.mkdir()
            outside_dir = Path(temp_dir) / "outside"
            outside_dir.mkdir()
            (outside_dir / "external-final_bundles.txt").write_text("bundles", encoding="utf-8")

            # Create symlink inside search_dir pointing outside
            symlink_run = search_dir / "symlink_run"
            try:
                os.symlink(outside_dir, symlink_run, target_is_directory=True)
            except OSError:
                self.skipTest("Symlinks not supported in environment")

            runs = discover_runs(search_dir)
            self.assertEqual(len(runs), 0)


class TestListRunFiles(unittest.TestCase):
    """Behavioral specifications for listing files in a run directory via `list_run_files`."""

    def test_should_list_text_files_with_names_and_sizes_when_run_directory_is_valid(self) -> None:
        """Verifies that all `.txt` files in a run directory are returned with names and positive sizes."""
        # Arrange
        runs = discover_runs(EXAMPLE_DATA_DIR)
        self.assertTrue(len(runs) > 0, "Precondition failed: No runs discovered in example_data")
        first_run_dir = EXAMPLE_DATA_DIR / runs[0]["id"]

        # Act
        files = list_run_files(first_run_dir)

        # Assert
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)
        final_bundles_found = False
        for f in files:
            self.assertIn("name", f)
            self.assertIn("size", f)
            self.assertTrue(f["name"].endswith(".txt"))
            self.assertIsInstance(f["size"], int)
            self.assertGreater(f["size"], 0)
            if "final_bundles.txt" in f["name"]:
                final_bundles_found = True
        self.assertTrue(final_bundles_found, "Expected *-final_bundles.txt in run files")

    def test_should_return_empty_list_when_directory_has_no_text_files(self) -> None:
        """Verifies that directories containing no `.txt` files return an empty list."""
        # Arrange
        with tempfile.TemporaryDirectory() as temp_dir:
            py_file = Path(temp_dir) / "test.py"
            py_file.write_text("print('hello')", encoding="utf-8")

            # Act
            files = list_run_files(temp_dir)

            # Assert
            self.assertEqual(files, [])

    def test_should_return_empty_list_or_raise_when_directory_does_not_exist(self) -> None:
        """Verifies behavior when the target run directory does not exist."""
        # Arrange
        nonexistent_dir = Path("/nonexistent/directory/for/testing/run/files")

        # Act & Assert
        try:
            files = list_run_files(nonexistent_dir)
            self.assertEqual(files, [])
        except FileNotFoundError:
            pass

    def test_should_return_empty_list_when_iterdir_raises_os_error(self) -> None:
        """Verifies list_run_files returns empty list when directory reading raises OSError."""
        with patch.object(Path, "iterdir", side_effect=OSError("permission denied")):
            self.assertEqual(list_run_files(Path(".")), [])


class TestResolveSourceFile(unittest.TestCase):
    """Behavioral specifications for multi-tier source code resolver `resolve_source_file`."""

    def setUp(self) -> None:
        """Sets up workspace root and search directory paths for resolution testing."""
        clear_source_cache()
        self.workspace_root = REPO_ROOT
        self.search_dir = EXAMPLE_DATA_DIR

    def tearDown(self) -> None:
        """Cleans up cache between test executions."""
        clear_source_cache()

    def test_should_resolve_file_when_exact_path_matches(self) -> None:
        """Tier 1: Verifies resolution when raw_path is an exact existing absolute path."""
        # Arrange
        exact_path = str(EXAMPLE_SOURCE_FILE.resolve())

        # Act
        result = resolve_source_file(exact_path, self.workspace_root, self.search_dir)

        # Assert
        self.assertTrue(result["found"])
        self.assertEqual(result["path"], exact_path)
        self.assertEqual(result["basename"], "bench_pcp_q_compute_micro.py")
        self.assertIn("def _make_micro_call", result["content"])

    def test_should_resolve_file_when_path_is_relative_to_workspace_root(self) -> None:
        """Tier 2: Verifies resolution when raw_path is relative to workspace_root."""
        # Arrange
        rel_path = "example_data/bench_pcp_q_compute_micro.py"

        # Act
        result = resolve_source_file(rel_path, self.workspace_root, self.search_dir)

        # Assert
        self.assertTrue(result["found"])
        self.assertEqual(result["basename"], "bench_pcp_q_compute_micro.py")
        self.assertIn("def _make_micro_call", result["content"])

    def test_should_resolve_file_when_path_is_relative_to_search_dir(self) -> None:
        """Tier 3: Verifies resolution when raw_path is relative to search_dir."""
        # Arrange
        search_dir = REPO_ROOT / "example_data"
        rel_path = "bench_pcp_q_compute_micro.py"

        # Act
        result = resolve_source_file(rel_path, self.workspace_root, search_dir)

        # Assert
        self.assertTrue(result["found"])
        self.assertEqual(result["basename"], "bench_pcp_q_compute_micro.py")
        self.assertIn("def _make_micro_call", result["content"])

    def test_should_resolve_file_when_only_basename_is_provided(self) -> None:
        """Tier 4: Verifies resolution via basename index search across workspace_root."""
        # Arrange
        remote_unmatched_path = "/build/src/deep/remote/path/bench_pcp_q_compute_micro.py"

        # Act
        result = resolve_source_file(remote_unmatched_path, self.workspace_root, self.search_dir)

        # Assert
        self.assertTrue(result["found"])
        self.assertEqual(result["basename"], "bench_pcp_q_compute_micro.py")
        self.assertIn("def _make_micro_call", result["content"])

    def test_should_return_not_found_when_file_does_not_exist(self) -> None:
        """Verifies that non-existent files return a standard not-found dictionary."""
        # Arrange
        nonexistent_path = "completely_nonexistent_file_xyz_123.py"

        # Act
        result = resolve_source_file(nonexistent_path, self.workspace_root, self.search_dir)

        # Assert
        self.assertFalse(result["found"])
        self.assertIn("error", result)

    def test_should_reject_file_when_size_exceeds_maximum_threshold(self) -> None:
        """Verifies that files exceeding the 100MB threshold are rejected."""
        # Arrange
        exact_path = str(EXAMPLE_SOURCE_FILE.resolve())

        # Act
        # Mock os.path.getsize to simulate file size exceeding 100 MB
        with patch("os.path.getsize", return_value=105 * 1024 * 1024):
            result = resolve_source_file(exact_path, self.workspace_root, self.search_dir)

        # Assert
        self.assertFalse(result["found"])
        self.assertIn("error", result)
        self.assertTrue(
            "exceed" in result["error"].lower() or "size" in result["error"].lower()
        )

    def test_should_reject_file_when_content_is_binary(self) -> None:
        """Verifies that binary files cannot be loaded as source text."""
        # Arrange
        with tempfile.NamedTemporaryFile(dir=self.workspace_root, suffix=".py", delete=False) as bin_file:
            bin_file.write(b"\x00\x01\x02\xff\xfe\x00")
            bin_path = bin_file.name

        try:
            # Act
            result = resolve_source_file(bin_path, self.workspace_root, self.search_dir)

            # Assert
            self.assertFalse(result["found"])
            self.assertIn("error", result)
            self.assertTrue(
                "binary" in result["error"].lower()
                or "decode" in result["error"].lower()
                or "text" in result["error"].lower()
            )
        finally:
            if os.path.exists(bin_path):
                os.unlink(bin_path)

    def test_should_return_error_when_raw_path_is_empty_or_whitespace(self) -> None:
        """Verifies resolve_source_file returns error when path is empty or whitespace."""
        res_empty = resolve_source_file("", self.workspace_root, self.search_dir)
        self.assertFalse(res_empty["found"])
        self.assertEqual(res_empty["error"], "File path cannot be empty")

        res_spaces = resolve_source_file("   ", self.workspace_root, self.search_dir)
        self.assertFalse(res_spaces["found"])
        self.assertEqual(res_spaces["error"], "File path cannot be empty")

    def test_should_return_none_when_search_basename_base_dir_is_not_directory(self) -> None:
        """Verifies _search_basename_in_tree returns None when base_dir does not exist."""
        from server import _search_basename_in_tree

        result = _search_basename_in_tree("/nonexistent/directory/xyz", "test.py")
        self.assertIsNone(result)

    def test_should_skip_directories_when_depth_exceeds_max_depth(self) -> None:
        """Verifies _search_basename_in_tree respects negative or exceeded max_depth."""
        from server import _search_basename_in_tree

        result = _search_basename_in_tree(self.workspace_root, "bench_pcp_q_compute_micro.py", max_depth=-1)
        self.assertIsNone(result)

    def test_should_resolve_file_from_search_dir_via_basename_index(self) -> None:
        """Tier 4: Verifies resolution from search_dir when file is not in workspace_root."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sd_file = tmp_path / "sd_only" / "unique_kernel.py"
            sd_file.parent.mkdir(parents=True, exist_ok=True)
            sd_file.write_text("def unique_kernel(): pass\n")

            empty_ws = tmp_path / "empty_ws"
            empty_ws.mkdir()

            result = resolve_source_file(
                "/remote/unmatched/path/unique_kernel.py",
                workspace_root=empty_ws,
                search_dir=sd_file.parent,
            )
            self.assertTrue(result["found"])
            self.assertEqual(result["basename"], "unique_kernel.py")
            self.assertIn("def unique_kernel", result["content"])

    def test_should_return_error_when_getsize_raises_os_error(self) -> None:
        """Verifies resolve_source_file returns error when inspecting file size fails."""
        with patch("os.path.getsize", side_effect=OSError("disk error")):
            result = resolve_source_file(
                "example_data/bench_pcp_q_compute_micro.py",
                self.workspace_root,
                self.search_dir,
            )
            self.assertFalse(result["found"])
            self.assertIn("Cannot inspect file size", result["error"])

    def test_should_return_error_when_open_raises_os_error(self) -> None:
        """Verifies resolve_source_file returns error when reading file fails with OSError."""
        with patch("builtins.open", side_effect=OSError("read error")):
            result = resolve_source_file(
                "example_data/bench_pcp_q_compute_micro.py",
                self.workspace_root,
                self.search_dir,
            )
            self.assertFalse(result["found"])
            self.assertIn("Cannot read file", result["error"])

    def test_should_reject_file_with_invalid_utf8_without_null_byte(self) -> None:
        """Verifies resolve_source_file rejects files with invalid UTF-8 bytes that have no nulls."""
        with tempfile.NamedTemporaryFile(dir=self.workspace_root, suffix=".py", delete=False) as bad_file:
            bad_file.write(b"\x80\x81\x82\x83")
            bad_path = bad_file.name
        try:
            result = resolve_source_file(bad_path, self.workspace_root, self.search_dir)
            self.assertFalse(result["found"])
            self.assertEqual(result["error"], "File is binary or not valid UTF-8 text")
        finally:
            if os.path.exists(bad_path):
                os.unlink(bad_path)

    def test_should_reject_with_access_denied_when_path_escapes_boundaries(self) -> None:
        """Verifies resolve_source_file rejects traversal attempts with access denied."""
        # Absolute path escape outside workspace and search roots
        result_abs = resolve_source_file("/etc/passwd", self.workspace_root, self.search_dir)
        self.assertFalse(result_abs["found"])
        self.assertIn("Access denied", result_abs.get("error", ""))

        # Relative traversal escape
        result_rel = resolve_source_file("../../../../etc/passwd", self.workspace_root, self.search_dir)
        self.assertFalse(result_rel["found"])
        self.assertIn("Access denied", result_rel.get("error", ""))

    def test_should_prune_excluded_directories_and_bound_walk_depth(self) -> None:
        """Verifies Tier 4 lookup ignores excluded directories like .git and respects depth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            ws = Path(temp_dir) / "workspace"
            ws.mkdir()
            git_dir = ws / ".git"
            git_dir.mkdir()
            (git_dir / "secret.py").write_text("print('secret')", encoding="utf-8")

            res = resolve_source_file("secret.py", ws, ws)
            self.assertFalse(res["found"])
            self.assertEqual(res.get("error"), "File not found")

    def test_should_cache_source_resolution_results(self) -> None:
        """Verifies repeated lookups use the bounded in-memory cache."""
        clear_source_cache()
        # Initial lookup
        res1 = resolve_source_file(
            "bench_pcp_q_compute_micro.py",
            self.workspace_root,
            self.search_dir,
        )
        self.assertTrue(res1["found"])

        # Second lookup should hit cache
        res2 = resolve_source_file(
            "bench_pcp_q_compute_micro.py",
            self.workspace_root,
            self.search_dir,
        )
        self.assertEqual(res1, res2)
        clear_source_cache()

    def test_should_evict_oldest_cache_entry_when_capacity_exceeded(self) -> None:
        """Verifies in-memory cache evicts oldest entry when exceeding capacity."""
        clear_source_cache()
        with patch("server.SOURCE_CACHE_CAPACITY", 2):
            res1 = resolve_source_file("dummy_file_1.py", self.workspace_root, self.search_dir)
            res2 = resolve_source_file("dummy_file_2.py", self.workspace_root, self.search_dir)
            res3 = resolve_source_file("dummy_file_3.py", self.workspace_root, self.search_dir)
            self.assertFalse(res1["found"])
            self.assertFalse(res2["found"])
            self.assertFalse(res3["found"])
        clear_source_cache()


class TestLLOServerHandler(unittest.TestCase):
    """Behavioral specifications for HTTP endpoints served by `LLOServerHandler`."""

    server: http.server.HTTPServer
    server_thread: threading.Thread
    host: str
    port: int
    base_url: str

    @classmethod
    def setUpClass(cls) -> None:
        """Spins up an ephemeral HTTP server on localhost port 0."""
        cls.web_root = REPO_ROOT
        cls.search_dir = EXAMPLE_DATA_DIR
        cls.workspace_root = REPO_ROOT

        cls.server = create_server(
            host="127.0.0.1",
            port=0,
            search_dir=cls.search_dir,
            workspace_root=cls.workspace_root,
            web_root=cls.web_root,
        )
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

        cls.host, cls.port = cls.server.server_address[:2]
        cls.base_url = f"http://{cls.host}:{cls.port}"

        discovered = discover_runs(cls.search_dir)
        cls.valid_run_id = discovered[0]["id"]
        run_files = list_run_files(cls.search_dir / cls.valid_run_id)
        cls.valid_bundle_file = run_files[0]["name"]

    @classmethod
    def tearDownClass(cls) -> None:
        """Shuts down and cleans up the ephemeral HTTP server."""
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=5.0)

    def _fetch(self, path: str) -> tuple[int, http.client.HTTPMessage, bytes]:
        """Sends a GET request to the ephemeral server.

        Args:
            path: Relative URL path and query parameters, e.g. "/api/status".

        Returns:
            tuple[int, HTTPMessage, bytes]: Response HTTP status code, headers, and raw bytes.
        """
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def _fetch_json(self, path: str) -> tuple[int, dict[str, Any]]:
        """Sends a GET request and parses the response body as JSON.

        Args:
            path: Relative URL path and query parameters.

        Returns:
            tuple[int, dict[str, Any]]: Response HTTP status code and parsed JSON data.
        """
        status, _, body = self._fetch(path)
        return status, json.loads(body.decode("utf-8"))

    def test_should_return_status_ok_when_querying_api_status(self) -> None:
        """Verifies `GET /api/status` returns HTTP 200 and server status metadata."""
        # Arrange & Act
        status, data = self._fetch_json("/api/status")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("mode"), "remote")
        self.assertEqual(data.get("searchDir"), str(self.search_dir))
        self.assertEqual(data.get("workspaceRoot"), str(self.workspace_root))

    def test_should_return_runs_list_when_querying_api_runs(self) -> None:
        """Verifies `GET /api/runs` returns HTTP 200 and list of discovered runs."""
        # Arrange & Act
        status, data = self._fetch_json("/api/runs")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertIn("runs", data)
        self.assertIsInstance(data["runs"], list)
        self.assertGreaterEqual(len(data["runs"]), 7)

        first_run = data["runs"][0]
        self.assertIn("id", first_run)
        self.assertIn("name", first_run)
        self.assertIn("bundleCount", first_run)
        self.assertIn("mtime", first_run)

    def test_should_return_files_list_when_querying_api_files_with_valid_run(self) -> None:
        """Verifies `GET /api/files?run=<valid_run>` returns HTTP 200 and run file list."""
        # Arrange
        _, runs_data = self._fetch_json("/api/runs")
        run_id = runs_data["runs"][0]["id"]

        # Act
        status, files_data = self._fetch_json(f"/api/files?run={urllib.parse.quote(run_id)}")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertEqual(files_data.get("runId"), run_id)
        self.assertIn("files", files_data)
        self.assertGreater(len(files_data["files"]), 0)

        first_file = files_data["files"][0]
        self.assertIn("name", first_file)
        self.assertIn("size", first_file)

    def test_should_return_forbidden_when_querying_api_files_with_directory_traversal(self) -> None:
        """Verifies `GET /api/files?run=../../invalid` returns HTTP 403 Forbidden."""
        # Arrange
        traversal_query = "/api/files?run=../../etc"

        # Act
        status, _, _ = self._fetch(traversal_query)

        # Assert
        self.assertEqual(status, http.HTTPStatus.FORBIDDEN)

    def test_should_return_file_content_when_querying_api_file_with_valid_run_and_filename(self) -> None:
        """Verifies `GET /api/file?run=...&filename=...` returns HTTP 200 with text content."""
        # Arrange
        _, runs_data = self._fetch_json("/api/runs")
        run_id = runs_data["runs"][0]["id"]
        _, files_data = self._fetch_json(f"/api/files?run={urllib.parse.quote(run_id)}")
        filename = files_data["files"][0]["name"]

        # Act
        status, headers, body = self._fetch(
            f"/api/file?run={urllib.parse.quote(run_id)}&filename={urllib.parse.quote(filename)}"
        )

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertIn("text/plain", headers.get("Content-Type", ""))
        self.assertGreater(len(body), 0)

    def test_should_return_byte_slice_when_offset_and_limit_are_specified(self) -> None:
        """Verifies `GET /api/file` with `offset` and `limit` returns the requested byte slice."""
        # Arrange
        _, runs_data = self._fetch_json("/api/runs")
        run_id = runs_data["runs"][0]["id"]
        _, files_data = self._fetch_json(f"/api/files?run={urllib.parse.quote(run_id)}")
        filename = files_data["files"][0]["name"]

        _, _, full_content = self._fetch(
            f"/api/file?run={urllib.parse.quote(run_id)}&filename={urllib.parse.quote(filename)}"
        )

        # Act
        status, _, slice_content = self._fetch(
            f"/api/file?run={urllib.parse.quote(run_id)}&filename={urllib.parse.quote(filename)}&offset=0&limit=50"
        )

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertEqual(slice_content, full_content[:50])

    def test_should_return_forbidden_when_querying_api_file_with_directory_traversal(self) -> None:
        """Verifies `GET /api/file` rejects directory traversal attempts with HTTP 403 Forbidden."""
        # Arrange
        traversal_query = "/api/file?run=pcp_q_compute_micro_llo&filename=../../etc/passwd"

        # Act
        status, _, _ = self._fetch(traversal_query)

        # Assert
        self.assertEqual(status, http.HTTPStatus.FORBIDDEN)

    def test_should_return_not_found_when_querying_api_file_with_missing_file(self) -> None:
        """Verifies `GET /api/file` returns HTTP 404 when requesting a non-existent file in a valid run."""
        # Arrange
        _, runs_data = self._fetch_json("/api/runs")
        run_id = runs_data["runs"][0]["id"]

        # Act
        status, _, _ = self._fetch(
            f"/api/file?run={urllib.parse.quote(run_id)}&filename=missing_file_xyz_123.txt"
        )

        # Assert
        self.assertEqual(status, http.HTTPStatus.NOT_FOUND)

    def test_should_return_source_content_when_querying_api_source_with_valid_path(self) -> None:
        """Verifies `GET /api/source?path=...` returns HTTP 200 and source content JSON."""
        # Arrange
        source_param = "bench_pcp_q_compute_micro.py"

        # Act
        status, data = self._fetch_json(f"/api/source?path={urllib.parse.quote(source_param)}")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertTrue(data.get("found"))
        self.assertEqual(data.get("basename"), "bench_pcp_q_compute_micro.py")
        self.assertIn("content", data)
        self.assertIn("def _make_micro_call", data["content"])

    def test_should_return_not_found_when_querying_api_source_with_missing_path(self) -> None:
        """Verifies `GET /api/source?path=...` returns HTTP 404 for non-existent source files."""
        # Arrange
        missing_param = "missing_source_xyz_987.py"

        # Act
        status, data = self._fetch_json(f"/api/source?path={urllib.parse.quote(missing_param)}")

        # Assert
        self.assertEqual(status, http.HTTPStatus.NOT_FOUND)
        self.assertFalse(data.get("found"))
        self.assertIn("error", data)

    def test_should_serve_index_html_when_requesting_root_path(self) -> None:
        """Verifies `GET /` serves `index.html` with HTTP 200."""
        # Arrange & Act
        status, headers, body = self._fetch("/")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn(b"TPU LLO Trace Viewer", body)

    def test_should_serve_index_html_when_requesting_index_html_explicitly(self) -> None:
        """Verifies `GET /index.html` serves `index.html` with HTTP 200."""
        # Arrange & Act
        status, headers, body = self._fetch("/index.html")

        # Assert
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn(b"TPU LLO Trace Viewer", body)

    def test_should_return_not_found_when_requesting_nonexistent_route(self) -> None:
        """Verifies arbitrary unknown routes return HTTP 404 Not Found."""
        # Arrange & Act
        status, _, _ = self._fetch("/api/nonexistent_route_abc_xyz")

        # Assert
        self.assertEqual(status, http.HTTPStatus.NOT_FOUND)

    def test_should_return_bad_request_when_run_param_missing_in_api_files(self) -> None:
        """Verifies GET /api/files returns HTTP 400 when 'run' parameter is missing."""
        status, data = self._fetch_json("/api/files")
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
        self.assertIn("error", data)

    def test_should_return_not_found_when_run_dir_does_not_exist_in_api_files(self) -> None:
        """Verifies GET /api/files?run=... returns HTTP 404 when run directory does not exist."""
        status, data = self._fetch_json("/api/files?run=nonexistent_run_folder_xyz")
        self.assertEqual(status, http.HTTPStatus.NOT_FOUND)
        self.assertIn("error", data)

    def test_should_return_bad_request_when_params_missing_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 400 when 'run' or 'filename' is missing."""
        status1, data1 = self._fetch_json("/api/file?run=pcp_q_compute_micro_llo")
        self.assertEqual(status1, http.HTTPStatus.BAD_REQUEST)

        status2, data2 = self._fetch_json("/api/file?filename=test.txt")
        self.assertEqual(status2, http.HTTPStatus.BAD_REQUEST)

    def test_should_return_bad_request_when_offset_or_limit_is_invalid_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 400 when offset or limit is not an integer."""
        url_bad_offset = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&offset=invalid"
        )
        status, data = self._fetch_json(url_bad_offset)
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)

        url_bad_limit = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&limit=invalid"
        )
        status, data = self._fetch_json(url_bad_limit)
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)

    def test_should_read_file_with_offset_and_no_limit_in_api_file(self) -> None:
        """Verifies GET /api/file works when offset is specified without limit."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&offset=10"
        )
        status, headers, body = self._fetch(url)
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertGreater(len(body), 0)

    def test_should_return_payload_too_large_when_file_exceeds_max_without_limit(self) -> None:
        """Verifies GET /api/file returns HTTP 413 when file exceeds 100 MB without limit."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}"
        )
        with patch("os.path.getsize", return_value=101 * 1024 * 1024):
            status, _, _ = self._fetch(url)
            self.assertEqual(status, http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    def test_should_return_server_error_when_getsize_fails_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 500 when os.path.getsize raises OSError."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}"
        )
        with patch("os.path.getsize", side_effect=OSError("stat error")):
            status, _, _ = self._fetch(url)
            self.assertEqual(status, http.HTTPStatus.INTERNAL_SERVER_ERROR)

    def test_should_return_server_error_when_file_read_fails_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 500 when opening file raises OSError."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}"
        )
        with patch("builtins.open", side_effect=OSError("disk read error")):
            status, _, _ = self._fetch(url)
            self.assertEqual(status, http.HTTPStatus.INTERNAL_SERVER_ERROR)

    def test_should_return_bad_request_when_path_is_missing_in_api_source(self) -> None:
        """Verifies GET /api/source returns HTTP 400 when 'path' query parameter is missing."""
        status, data = self._fetch_json("/api/source")
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)
        self.assertFalse(data.get("found"))

    def test_should_return_403_when_api_source_path_escapes_boundaries(self) -> None:
        """Verifies GET /api/source returns HTTP 403 Forbidden when path escapes boundaries."""
        status_abs, data_abs = self._fetch_json("/api/source?path=/etc/passwd")
        self.assertEqual(status_abs, http.HTTPStatus.FORBIDDEN)
        self.assertFalse(data_abs.get("found"))
        self.assertIn("Access denied", data_abs.get("error", ""))

        status_rel, data_rel = self._fetch_json("/api/source?path=../../../../etc/passwd")
        self.assertEqual(status_rel, http.HTTPStatus.FORBIDDEN)
        self.assertFalse(data_rel.get("found"))
        self.assertIn("Access denied", data_rel.get("error", ""))

    def test_should_return_400_when_offset_is_negative_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 400 when offset < 0."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&offset=-5"
        )
        status, _ = self._fetch_json(url)
        self.assertEqual(status, http.HTTPStatus.BAD_REQUEST)

    def test_should_return_400_when_limit_is_zero_or_negative_in_api_file(self) -> None:
        """Verifies GET /api/file returns HTTP 400 when limit <= 0."""
        url_zero = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&limit=0"
        )
        status_zero, _ = self._fetch_json(url_zero)
        self.assertEqual(status_zero, http.HTTPStatus.BAD_REQUEST)

        url_neg = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&limit=-10"
        )
        status_neg, _ = self._fetch_json(url_neg)
        self.assertEqual(status_neg, http.HTTPStatus.BAD_REQUEST)

    def test_should_cap_slice_limit_to_max_file_size(self) -> None:
        """Verifies GET /api/file caps excessively large limit parameter."""
        url = (
            f"/api/file?run={urllib.parse.quote(self.valid_run_id)}"
            f"&filename={urllib.parse.quote(self.valid_bundle_file)}&limit=999999999"
        )
        status, headers, body = self._fetch(url)
        self.assertEqual(status, http.HTTPStatus.OK)
        self.assertGreater(len(body), 0)

    def test_should_fallback_handler_properties_when_server_missing_attributes(self) -> None:
        """Verifies fallback properties when handler server does not set attributes."""
        mock_server = MagicMock(spec=[])
        with patch("http.server.SimpleHTTPRequestHandler.__init__"):
            handler = LLOServerHandler.__new__(LLOServerHandler)
            handler.server = mock_server
            self.assertEqual(handler.search_dir, Path("."))
            self.assertEqual(handler.workspace_root, Path("."))
            self.assertEqual(handler.max_depth, 6)
            self.assertTrue(handler.web_root.is_dir())


class TestMainAndCLI(unittest.TestCase):
    """Unit tests for CLI main entrypoint and keyboard shutdown."""

    def test_should_show_help_when_passed_help_flag(self) -> None:
        """Verifies main(["--help"]) outputs help text and exits with code 0."""
        with self.assertRaises(SystemExit) as cm:
            with patch("sys.stdout"):
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_should_start_server_and_handle_keyboard_interrupt(self) -> None:
        """Verifies main starts server and handles KeyboardInterrupt cleanly."""
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt

        with patch("server.create_server", return_value=mock_server):
            with patch("sys.stdout"):
                main(["--port", "8080", "--host", "127.0.0.1"])

        mock_server.serve_forever.assert_called_once()
        mock_server.server_close.assert_called_once()

    def test_should_execute_as_main_module(self) -> None:
        """Verifies running module as __main__ invokes main entrypoint."""
        with patch("sys.argv", ["server.py", "--help"]):
            with patch("sys.stdout"):
                with self.assertRaises(SystemExit) as cm:
                    runpy.run_module("server", run_name="__main__")
                self.assertEqual(cm.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
