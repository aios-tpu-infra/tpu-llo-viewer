"""HTTP server backend for the remote TPU LLO trace viewer.

This module provides a multi-threaded HTTP server that serves the static LLO
viewer frontend and exposes REST API endpoints for discovering LLO dump runs,
listing run files, streaming file slices, and resolving source code files across
workspace and search roots with strict path traversal guards.
"""

from __future__ import annotations

import argparse
import http
import http.server
import json
import os
from pathlib import Path
import socketserver
import sys
from typing import Any
import urllib.parse

MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100 MB
MAX_SOURCE_SEARCH_DEPTH: int = 4
SOURCE_CACHE_CAPACITY: int = 500
EXCLUDED_SOURCE_DIRS: frozenset[str] = frozenset({
    ".git",
    ".venv",
    "venv",
    "cache",
    "results",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".worktrees",
})

_SOURCE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def clear_source_cache() -> None:
    """Clears the in-memory source resolution cache (used in tests and reset flows)."""
    _SOURCE_CACHE.clear()



def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for the LLO trace viewer server.

    Args:
        args: Optional list of CLI argument strings. When None, arguments are read
            from sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed CLI namespace with attributes:
            - dir (str): Search directory for LLO runs (default: ".").
            - port (int): TCP port to bind (default: 8080).
            - host (str): Host address to bind (default: "0.0.0.0").
            - workspace_root (str): Workspace root for source code resolution (default: ".").
            - max_depth (int): Maximum directory depth for run discovery (default: 6).
    """
    parser = argparse.ArgumentParser(
        description="Remote HTTP server for TPU LLO trace viewer."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory to search for LLO run dumps (default: .)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=".",
        help="Workspace root for source code lookup (default: .)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum recursion depth for run discovery (default: 6)",
    )
    return parser.parse_args(args)


def is_safe_path(base_dir: str | Path, target_path: str | Path) -> bool:
    """Validates that target_path resides within base_dir to prevent directory traversal.

    Both paths are resolved using canonical realpaths with symlink resolution.
    Path containment is verified using os.path.commonpath to prevent prefix collisions.

    Args:
        base_dir: Boundary directory that target_path must reside within.
        target_path: Candidate file or directory path to check.

    Returns:
        bool: True if target_path is inside base_dir; False if it escapes or
            resolves outside base_dir.
    """
    try:
        base_resolved = os.path.realpath(str(base_dir))
        target_resolved = os.path.realpath(str(target_path))
        return os.path.commonpath([base_resolved, target_resolved]) == base_resolved
    except (ValueError, OSError):
        return False


def discover_runs(
    search_dir: str | Path,
    max_depth: int = 6,
) -> list[dict[str, Any]]:
    """Recursively scans search_dir up to max_depth for LLO runs containing final bundles.

    A directory is considered an LLO run directory if it contains at least one
    file ending with `*-final_bundles.txt`.

    Args:
        search_dir: Root directory path from which to begin discovery.
        max_depth: Maximum recursion depth relative to search_dir (default: 6).

    Returns:
        list[dict[str, Any]]: Discovered runs sorted by mtime descending. Each entry contains:
            - id (str): Relative directory path from search_dir.
            - name (str): Relative directory path from search_dir.
            - bundleCount (int): Number of final bundle files found.
            - mtime (float): Latest modification timestamp among the bundle files.
            Returns an empty list if search_dir does not exist or contains no runs.
    """
    search_path = Path(search_dir)
    if not search_path.is_dir():
        return []

    runs: list[dict[str, Any]] = []

    for root, dirs, files in os.walk(search_path, followlinks=True):
        rel_root = os.path.relpath(root, search_path)
        depth = 0 if rel_root == "." else len(Path(rel_root).parts)

        # Prune symlinks or subdirectories pointing outside search_path
        dirs[:] = [
            d
            for d in dirs
            if is_safe_path(search_path, os.path.join(root, d))
        ]

        if depth >= max_depth:
            dirs.clear()
        if depth > max_depth:
            continue

        bundle_files = [
            f
            for f in files
            if f.endswith("-final_bundles.txt")
            and is_safe_path(search_path, os.path.join(root, f))
        ]
        if bundle_files:
            mtimes: list[float] = []
            for f in bundle_files:
                try:
                    mtimes.append(os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    continue
            if mtimes:
                latest_mtime = max(mtimes)
            else:
                try:
                    latest_mtime = os.path.getmtime(root)
                except OSError:
                    latest_mtime = 0.0

            rel_id = rel_root.replace("\\", "/")
            runs.append({
                "id": rel_id,
                "name": rel_id,
                "bundleCount": len(bundle_files),
                "mtime": latest_mtime,
            })

    runs.sort(key=lambda r: (r["mtime"], r["id"]), reverse=True)
    return runs


def list_run_files(run_dir: str | Path) -> list[dict[str, Any]]:
    """Lists all text files in a specified run directory.

    Args:
        run_dir: Path to the target run directory.

    Returns:
        list[dict[str, Any]]: Alphabetically sorted list of file metadata dicts:
            - name (str): File name ending in .txt.
            - size (int): File size in bytes.
            Returns an empty list if run_dir does not exist or has no .txt files.
    """
    run_path = Path(run_dir)
    if not run_path.is_dir():
        return []

    results: list[dict[str, Any]] = []
    try:
        for entry in run_path.iterdir():
            if entry.is_file() and entry.name.endswith(".txt"):
                results.append({
                    "name": entry.name,
                    "size": entry.stat().st_size,
                })
    except OSError:
        return []

    results.sort(key=lambda item: item["name"])
    return results


def _search_basename_in_tree(
    base_dir: str | Path,
    target_basename: str,
    max_depth: int = MAX_SOURCE_SEARCH_DEPTH,
) -> Path | None:
    """Searches for a file by basename within a directory tree, bounded by max_depth.

    Excludes repository internals and virtual environment directories.

    Args:
        base_dir: Root directory of tree to search.
        target_basename: File name to locate.
        max_depth: Maximum recursion depth relative to base_dir.

    Returns:
        Path | None: Resolved Path if found and safe, or None.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        return None

    for root, dirs, files in os.walk(base_path, followlinks=False):
        rel_root = os.path.relpath(root, base_path)
        depth = 0 if rel_root == "." else len(Path(rel_root).parts)

        # Prune excluded and hidden directories
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDED_SOURCE_DIRS and not d.startswith(".")
        ]

        if depth >= max_depth:
            dirs.clear()
        if depth > max_depth:
            continue

        if target_basename in files:
            candidate = Path(root) / target_basename
            if candidate.is_file() and is_safe_path(base_path, candidate):
                return candidate

    return None


def _cache_source_result(key: tuple[str, str, str], result: dict[str, Any]) -> None:
    """Stores result in bounded in-memory cache, evicting oldest item when full.

    Args:
        key: Tuple of (clean_raw_path, resolved_workspace_root, resolved_search_dir).
        result: Resolution result dictionary to cache.
    """
    if len(_SOURCE_CACHE) >= SOURCE_CACHE_CAPACITY:
        oldest_key = next(iter(_SOURCE_CACHE), None)
        if oldest_key is not None:
            _SOURCE_CACHE.pop(oldest_key, None)
    _SOURCE_CACHE[key] = result


def resolve_source_file(
    raw_path: str,
    workspace_root: str | Path,
    search_dir: str | Path,
) -> dict[str, Any]:
    """Resolves source code files across workspace root and search directory.

    Applies a 4-tier lookup strategy:
    - Tier 1: Exact absolute or relative path match if within allowed boundaries.
    - Tier 2: Path relative to workspace_root.
    - Tier 3: Path relative to search_dir.
    - Tier 4: Basename index lookup walking workspace_root, then search_dir (depth-bounded).

    Enforces guards:
    - Path traversal containment: rejects candidate paths escaping workspace and search boundaries.
    - Bounded in-memory caching to eliminate redundant disk walks.
    - Maximum file size of 100 MB.
    - Valid UTF-8 text validation (rejects binary files containing null bytes).

    Args:
        raw_path: Source file path or basename string.
        workspace_root: Root workspace directory for resolving source code.
        search_dir: Search directory containing LLO run dumps.

    Returns:
        dict[str, Any]: If resolved successfully:
            {
                "found": True,
                "path": str,
                "basename": str,
                "content": str,
            }
            If unresolved, access denied, or invalid:
            {
                "found": False,
                "error": str,
            }
    """
    clean_raw = raw_path.strip()
    if not clean_raw:
        return {"found": False, "error": "File path cannot be empty"}

    cache_key = (
        clean_raw,
        os.path.realpath(str(workspace_root)),
        os.path.realpath(str(search_dir)),
    )
    if cache_key in _SOURCE_CACHE:
        return _SOURCE_CACHE[cache_key]

    if ".." in Path(clean_raw).parts:
        candidate_ws = Path(workspace_root) / clean_raw.lstrip("/\\")
        candidate_sd = Path(search_dir) / clean_raw.lstrip("/\\")
        if not (is_safe_path(workspace_root, candidate_ws) or is_safe_path(search_dir, candidate_sd)):
            denied_result = {
                "found": False,
                "error": "Access denied: Path outside allowed workspace and search boundaries",
            }
            _cache_source_result(cache_key, denied_result)
            return denied_result

    candidate_path: Path | None = None
    target_basename = os.path.basename(clean_raw)

    # Tier 1: exact path
    if os.path.isfile(clean_raw):
        candidate_path = Path(clean_raw)
    else:
        # Tier 2: relative to workspace_root
        normalized_rel = clean_raw.lstrip("/\\")
        candidate_ws = Path(workspace_root) / normalized_rel
        if candidate_ws.is_file():
            candidate_path = candidate_ws
        else:
            # Tier 3: relative to search_dir
            candidate_sd = Path(search_dir) / normalized_rel
            if candidate_sd.is_file():
                candidate_path = candidate_sd
            elif target_basename:
                # Tier 4: Basename index lookup across workspace_root, then search_dir
                candidate_path = _search_basename_in_tree(
                    workspace_root, target_basename, max_depth=MAX_SOURCE_SEARCH_DEPTH
                )
                if candidate_path is None:
                    candidate_path = _search_basename_in_tree(
                        search_dir, target_basename, max_depth=MAX_SOURCE_SEARCH_DEPTH
                    )

    if candidate_path is None or not candidate_path.is_file():
        result = {"found": False, "error": "File not found"}
        _cache_source_result(cache_key, result)
        return result

    if not (is_safe_path(workspace_root, candidate_path) or is_safe_path(search_dir, candidate_path)):
        result = {
            "found": False,
            "error": "Access denied: Path outside allowed workspace and search boundaries",
        }
        _cache_source_result(cache_key, result)
        return result

    try:
        size = os.path.getsize(candidate_path)
    except OSError as err:
        result = {"found": False, "error": f"Cannot inspect file size: {err}"}
        _cache_source_result(cache_key, result)
        return result

    if size > MAX_FILE_SIZE:
        result = {
            "found": False,
            "error": "File size exceeds maximum threshold (100 MB)",
        }
        _cache_source_result(cache_key, result)
        return result

    try:
        with open(candidate_path, "rb") as f:
            raw_bytes = f.read()
    except OSError as err:
        result = {"found": False, "error": f"Cannot read file: {err}"}
        _cache_source_result(cache_key, result)
        return result

    if b"\x00" in raw_bytes:
        result = {
            "found": False,
            "error": "File is binary or not valid UTF-8 text",
        }
        _cache_source_result(cache_key, result)
        return result

    try:
        text_content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        result = {
            "found": False,
            "error": "File is binary or not valid UTF-8 text",
        }
        _cache_source_result(cache_key, result)
        return result

    result = {
        "found": True,
        "path": str(candidate_path.resolve()),
        "basename": candidate_path.name,
        "content": text_content,
    }
    _cache_source_result(cache_key, result)
    return result


class LLOServer(http.server.ThreadingHTTPServer):
    """Multi-threaded HTTP server holding LLO viewer configuration."""

    search_dir: Path
    workspace_root: Path
    web_root: Path
    max_depth: int

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[http.server.BaseHTTPRequestHandler],
        search_dir: Path,
        workspace_root: Path,
        web_root: Path,
        max_depth: int,
    ) -> None:
        """Initializes the LLOServer with configuration parameters.

        Args:
            server_address: (host, port) network tuple.
            RequestHandlerClass: HTTP request handler class.
            search_dir: Path to LLO run search directory.
            workspace_root: Path to workspace root directory.
            web_root: Path to web static root directory.
            max_depth: Maximum recursion depth for run discovery.
        """
        self.search_dir = search_dir
        self.workspace_root = workspace_root
        self.web_root = web_root
        self.max_depth = max_depth
        super().__init__(server_address, RequestHandlerClass)


class LLOServerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler serving LLO REST API endpoints and static assets."""

    @property
    def search_dir(self) -> Path:
        """Returns the search directory from the parent server or default."""
        return getattr(self.server, "search_dir", Path("."))

    @property
    def workspace_root(self) -> Path:
        """Returns the workspace root from the parent server or default."""
        return getattr(self.server, "workspace_root", Path("."))

    @property
    def web_root(self) -> Path:
        """Returns the web static root directory from the parent server or default."""
        return getattr(self.server, "web_root", Path(__file__).resolve().parent)

    @property
    def max_depth(self) -> int:
        """Returns the maximum search depth from the parent server or default."""
        return getattr(self.server, "max_depth", 6)

    def __init__(
        self,
        request: Any,
        client_address: Any,
        server: socketserver.BaseServer,
        directory: str | None = None,
    ) -> None:
        """Initializes the handler with the web root as the static directory."""
        web_dir = getattr(server, "web_root", Path(__file__).resolve().parent)
        super().__init__(
            request,
            client_address,
            server,
            directory=str(web_dir) if directory is None else directory,
        )

    def translate_path(self, path: str) -> str:
        """Translates a URL path to the local filesystem path within web_root.

        Args:
            path: Relative or absolute URL request path.

        Returns:
            str: Resolved local filesystem path within web_root.
        """
        self.directory = str(self.web_root)
        return super().translate_path(path)

    def _send_json(self, status: http.HTTPStatus, data: dict[str, Any]) -> None:
        """Sends a JSON HTTP response.

        Args:
            status: HTTPStatus response code.
            data: Dictionary payload to serialize as JSON.
        """
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Handles HTTP GET requests for API endpoints and static viewer assets."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/status":
            self._handle_api_status()
        elif path == "/api/runs":
            self._handle_api_runs()
        elif path == "/api/files":
            self._handle_api_files(params)
        elif path == "/api/file":
            self._handle_api_file(params)
        elif path == "/api/source":
            self._handle_api_source(params)
        elif path.startswith("/api/"):
            self._send_json(
                http.HTTPStatus.NOT_FOUND,
                {"error": f"API endpoint '{path}' not found"},
            )
        else:
            super().do_GET()

    def _handle_api_status(self) -> None:
        """Serves GET /api/status with server runtime configuration."""
        self._send_json(
            http.HTTPStatus.OK,
            {
                "status": "ok",
                "mode": "remote",
                "searchDir": str(self.search_dir),
                "workspaceRoot": str(self.workspace_root),
            },
        )

    def _handle_api_runs(self) -> None:
        """Serves GET /api/runs with list of discovered LLO runs."""
        runs = discover_runs(self.search_dir, self.max_depth)
        self._send_json(http.HTTPStatus.OK, {"runs": runs})

    def _handle_api_files(self, params: dict[str, list[str]]) -> None:
        """Serves GET /api/files?run=<run_id>."""
        run_ids = params.get("run", [])
        if not run_ids or not run_ids[0]:
            self._send_json(
                http.HTTPStatus.BAD_REQUEST,
                {"error": "Missing 'run' query parameter"},
            )
            return

        run_param = run_ids[0]
        run_dir = os.path.join(self.search_dir, run_param)

        if not is_safe_path(self.search_dir, run_dir):
            self._send_json(
                http.HTTPStatus.FORBIDDEN,
                {"error": "Forbidden: Path traversal detected"},
            )
            return

        if not os.path.isdir(run_dir):
            self._send_json(
                http.HTTPStatus.NOT_FOUND,
                {"error": f"Run directory '{run_param}' not found"},
            )
            return

        files = list_run_files(run_dir)
        self._send_json(
            http.HTTPStatus.OK,
            {"runId": run_param, "files": files},
        )

    def _handle_api_file(self, params: dict[str, list[str]]) -> None:
        """Serves GET /api/file?run=<run_id>&filename=<filename>[&offset=&limit=]."""
        run_ids = params.get("run", [])
        filenames = params.get("filename", [])
        if not run_ids or not run_ids[0] or not filenames or not filenames[0]:
            self._send_json(
                http.HTTPStatus.BAD_REQUEST,
                {"error": "Missing 'run' or 'filename' query parameter"},
            )
            return

        run_param = run_ids[0]
        filename_param = filenames[0]
        file_path = os.path.join(self.search_dir, run_param, filename_param)

        if not is_safe_path(self.search_dir, file_path):
            self._send_json(
                http.HTTPStatus.FORBIDDEN,
                {"error": "Forbidden: Path traversal detected"},
            )
            return

        if not os.path.isfile(file_path):
            self._send_json(
                http.HTTPStatus.NOT_FOUND,
                {"error": f"File '{filename_param}' not found in run '{run_param}'"},
            )
            return

        offset = 0
        limit: int | None = None
        if "offset" in params and params["offset"]:
            try:
                offset_val = int(params["offset"][0])
                if offset_val < 0:
                    self._send_json(
                        http.HTTPStatus.BAD_REQUEST,
                        {"error": "Parameter 'offset' must be non-negative"},
                    )
                    return
                offset = offset_val
            except ValueError:
                self._send_json(
                    http.HTTPStatus.BAD_REQUEST,
                    {"error": "Invalid 'offset' parameter"},
                )
                return

        if "limit" in params and params["limit"]:
            try:
                limit_val = int(params["limit"][0])
                if limit_val <= 0:
                    self._send_json(
                        http.HTTPStatus.BAD_REQUEST,
                        {"error": "Parameter 'limit' must be a positive integer"},
                    )
                    return
                limit = min(limit_val, MAX_FILE_SIZE)
            except ValueError:
                self._send_json(
                    http.HTTPStatus.BAD_REQUEST,
                    {"error": "Invalid 'limit' parameter"},
                )
                return

        try:
            file_size = os.path.getsize(file_path)
        except OSError as err:
            self._send_json(
                http.HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Cannot inspect file size: {err}"},
            )
            return

        if limit is None and file_size > MAX_FILE_SIZE:
            self._send_json(
                http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "File size exceeds 100 MB limit and no slice limit specified"},
            )
            return

        try:
            with open(file_path, "rb") as f:
                if offset > 0:
                    f.seek(offset)
                if limit is not None:
                    data = f.read(limit)
                else:
                    data = f.read()
        except OSError as err:
            self._send_json(
                http.HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"Cannot read file: {err}"},
            )
            return

        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api_source(self, params: dict[str, list[str]]) -> None:
        """Serves GET /api/source?path=<path_or_basename>."""
        paths = params.get("path", [])
        if not paths or not paths[0]:
            self._send_json(
                http.HTTPStatus.BAD_REQUEST,
                {"found": False, "error": "Missing 'path' query parameter"},
            )
            return

        raw_path = paths[0]
        result = resolve_source_file(raw_path, self.workspace_root, self.search_dir)
        if result.get("found"):
            status = http.HTTPStatus.OK
        elif result.get("error", "").startswith("Access denied"):
            status = http.HTTPStatus.FORBIDDEN
        else:
            status = http.HTTPStatus.NOT_FOUND
        self._send_json(status, result)


def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    search_dir: str | Path = ".",
    workspace_root: str | Path = ".",
    web_root: str | Path | None = None,
    max_depth: int = 6,
) -> http.server.HTTPServer:
    """Creates and configures an LLOServer instance.

    Args:
        host: Host IP or hostname to bind (default: "0.0.0.0").
        port: TCP port to listen on (default: 8080).
        search_dir: Root directory for LLO runs (default: ".").
        workspace_root: Root directory for source code resolution (default: ".").
        web_root: Root directory for static viewer files (default: directory containing server.py).
        max_depth: Maximum directory traversal depth (default: 6).

    Returns:
        http.server.HTTPServer: Initialized LLOServer instance.
    """
    resolved_web_root = (
        Path(web_root).resolve()
        if web_root is not None
        else Path(__file__).resolve().parent
    )
    resolved_search_dir = Path(search_dir)
    resolved_workspace_root = Path(workspace_root)

    return LLOServer(
        (host, port),
        LLOServerHandler,
        search_dir=resolved_search_dir,
        workspace_root=resolved_workspace_root,
        web_root=resolved_web_root,
        max_depth=max_depth,
    )


def main(cli_args: list[str] | None = None) -> None:
    """CLI entrypoint for running the TPU LLO trace viewer server.

    Args:
        cli_args: Optional CLI argument list. Defaults to sys.argv[1:].
    """
    args = parse_args(cli_args)
    search_dir = Path(args.dir).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    web_root = Path(__file__).resolve().parent

    server = create_server(
        host=args.host,
        port=args.port,
        search_dir=search_dir,
        workspace_root=workspace_root,
        web_root=web_root,
        max_depth=args.max_depth,
    )
    print(f"Serving TPU LLO Viewer on http://{args.host}:{args.port}")
    print(f"Search directory: {search_dir}")
    print(f"Workspace root: {workspace_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
