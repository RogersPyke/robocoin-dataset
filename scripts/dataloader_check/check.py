"""Unified dataloader checker CLI with local and distributed modes.

Three modes:
  - local: Sequential checking in a single process
  - server: Distributed server that assigns tasks to clients
  - client: Client processes that connect to the server

Usage:
  python scripts/dataloader_check/check.py local --config_path ./db/postgresql_config.yaml
  python scripts/dataloader_check/check.py server --config_path ./db/postgresql_config.yaml --host 0.0.0.0 --port 2010
  python scripts/dataloader_check/check.py client --host 172.16.13.140 --port 2010 --num_clients 4

Run `python scripts/dataloader_check/dataloader_check.py <mode> --help` for detailed options.
"""

import argparse
import asyncio
import logging
import multiprocessing as mp
from pathlib import Path

from robocoin_dataset.dataloader_check.client import DataLoaderCheckerClient
from robocoin_dataset.dataloader_check.local import DataLoaderChecker
from robocoin_dataset.dataloader_check.server import DataLoaderCheckerServer
from robocoin_dataset.utils.logger import setup_logger


# ============================================================================
# Local Mode
# ============================================================================


def run_local(args) -> None:
    """Run local sequential dataloader checker."""
    logger = setup_logger(
        name="dataloader_check",
        log_dir=Path(args.log_dir),
        level=logging.INFO,
    )

    checker = DataLoaderChecker(
        args.config_path,
        logger=logger,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
    )
    checker.check_one_repo()


# ============================================================================
# Server Mode
# ============================================================================


async def run_server(args) -> None:
    """Run distributed dataloader checker server."""
    config_path = Path(args.config_path).expanduser().absolute()

    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        exit(1)

    logger = setup_logger(
        name="dataloader_check_server",
        log_dir=Path(args.log_dir),
        level=logging.INFO,
    )

    checker = DataLoaderCheckerServer(
        db_file_path=config_path,
        host=args.host,
        port=args.port,
        logger=logger,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        target_dataset_uuid=args.target_dataset_uuid,
    )

    await checker.start()


# ============================================================================
# Client Mode
# ============================================================================


async def run_client_process(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """Async client logic executed by each process."""
    logger = setup_logger(
        name=f"dataloader_check_client{process_id}",
        log_dir=log_path,
        level=logging.ERROR,
    )

    client = DataLoaderCheckerClient(
        server_uri=server_uri,
        heartbeat_interval=heartbeat_interval,
        logger=logger,
    )
    await client.run()


def client_process_main(
    server_uri: str,
    heartbeat_interval: float,
    log_path: str,
    process_id: int,
) -> None:
    """Multiprocessing entry function for client."""
    asyncio.run(
        run_client_process(
            server_uri=server_uri,
            heartbeat_interval=heartbeat_interval,
            log_path=log_path,
            process_id=process_id,
        )
    )


def run_client(args) -> None:
    """Run distributed dataloader checker clients."""
    server_uri = f"ws://{args.host}:{args.port}"
    num_clients = max(1, min(args.num_clients, 8))

    processes = []
    for i in range(num_clients):
        proc = mp.Process(
            target=client_process_main,
            kwargs=dict(
                server_uri=server_uri,
                heartbeat_interval=args.heartbeat_interval,
                log_path=args.log_dir,
                process_id=i,
            ),
        )
        proc.start()
        processes.append(proc)

    print(f"Started {num_clients} client processes. Waiting for them to finish...")

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        for proc in processes:
            proc.terminate()
            proc.join(timeout=2)


# ============================================================================
# Argument Parsing and Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point with subcommand-based argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified dataloader checker for PostgreSQL datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local mode
  python scripts/dataloader_check/dataloader_check.py local \\
    --config_path ./db/postgresql_config.yaml \\
    --log_dir ./logs/dataloader_check \\
    --num_workers 8 \\
    --sample_rate 0.1

  # Server mode
  python scripts/dataloader_check/dataloader_check.py server \\
    --config_path ./db/postgresql_config.yaml \\
    --host 0.0.0.0 \\
    --port 2010 \\
    --log_dir ./logs/dataloader_check_server \\
    --num_workers 8 \\
    --sample_rate 0.1

  # Client mode
  python scripts/dataloader_check/dataloader_check.py client \\
    --host 172.16.13.140 \\
    --port 2010 \\
    --log_dir ./logs/dataloader_check_client \\
    --num_clients 4 \\
    --heartbeat-interval 10.0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ====== LOCAL SUBCOMMAND ======
    local_parser = subparsers.add_parser(
        "local", help="Run sequential dataloader checker locally"
    )
    local_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to PostgreSQL configuration file (YAML format)",
    )
    local_parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )
    local_parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader",
    )
    local_parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="Sample rate for dataset checking (0.0-1.0)",
    )

    # ====== SERVER SUBCOMMAND ======
    server_parser = subparsers.add_parser(
        "server", help="Run distributed dataloader checker server"
    )
    server_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to PostgreSQL configuration file (YAML format)",
    )
    server_parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=2010,
        help="Port to run the server",
    )
    server_parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloader",
    )
    server_parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="Sample rate for dataset checking (0.0-1.0)",
    )
    server_parser.add_argument(
        "--target_dataset_uuid",
        type=str,
        default="",
        help="Target dataset UUID to check (if empty, check all eligible datasets)",
    )

    # ====== CLIENT SUBCOMMAND ======
    client_parser = subparsers.add_parser(
        "client", help="Run distributed dataloader checker clients"
    )
    client_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host to connect to",
    )
    client_parser.add_argument(
        "--port",
        type=int,
        default=2010,
        help="Server port to connect to",
    )
    client_parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Path to the log directory",
    )
    client_parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=10.0,
        help="Heartbeat interval for each client",
    )
    client_parser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of concurrent client processes to spawn",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "local":
        run_local(args)
    elif args.command == "server":
        asyncio.run(run_server(args))
    elif args.command == "client":
        mp.set_start_method("spawn", force=True)
        run_client(args)


if __name__ == "__main__":
    main()
