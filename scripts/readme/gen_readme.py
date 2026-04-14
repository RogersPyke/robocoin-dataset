"""
README CLI entry script.

Purpose:
    Provide a command-line interface for generating dataset README files
    from existing metadata `info.yaml`.

Dependencies:
    - argparse: Parse command-line arguments
    - pathlib: Path normalization
    - traceback: Error stack capture
    - robocoin_dataset.readme.gen_readme: Core README generator class
    - robocoin_dataset.readme._logging: Logger setup utility
    - robocoin_dataset.utils.log_config: Colored success/error logs

Usage examples:
    1) Generate README from existing info.yaml:
       python scripts/readme/gen_readme.py --dataset-path /data/my_dataset

Input:
    Command-line arguments only.

Output:
    - On success: print generated README path to stdout, return exit code 0.
    - On failure: print error to stderr via logger, return non-zero exit code.
    - Audit logs are written to <project_root>/logs/gen_readme/gen_readme_cli_<YYYYMMDDHHMMSS>.log.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

from robocoin_dataset.database.database import DatasetDatabase
from robocoin_dataset.readme.gen_readme import ReadmeGenerator
from robocoin_dataset.readme._logging import setup_readme_logger
from robocoin_dataset.readme.task import list_readme_tasks
from robocoin_dataset.utils.log_config import log_error, log_success


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments for README generation.

    Input:
        argv (Optional[list[str]]): Argument list. If None, argparse reads sys.argv.

    Output:
        argparse.Namespace: Parsed arguments object with fields:
            - dataset_path (Optional[str])
            - db_cfg_path (Optional[str])
            - ignore_uploaded (bool)
            - target_dataset_uuid (str)

    Scenario:
        Called by main() as the only CLI input entry.

    """
    parser = argparse.ArgumentParser(
        prog="gen_readme.py",
        description="Generate dataset README.md from YAML metadata and Jinja2 template.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Dataset/hardlink directory path (root directory where data is stored).",
    )
    parser.add_argument(
        "--db-cfg-path",
        default=None,
        help="PostgreSQL config YAML path for batch README generation mode.",
    )
    parser.add_argument(
        "--ignore-uploaded",
        action="store_true",
        help="In batch mode, include uploaded datasets too.",
    )
    parser.add_argument(
        "--target-dataset-uuid",
        default="",
        help="Only process this dataset UUID in batch mode.",
    )
    return parser.parse_args(argv)


def get_project_root() -> Path:
    """
    Resolve repository root path from current script location.

    Input:
        None.

    Output:
        Path: Absolute project root path.

    Scenario:
        Used to build fixed template/log paths.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent.parent


def create_cli_logger(project_root: Path) -> logging.Logger:
    """
    Create a dedicated CLI logger for traceable audit logs.

    Input:
        project_root (Path): Absolute project root path.

    Output:
        logging.Logger: Configured logger with file + console handlers.

    Scenario:
        Called before running generation so all CLI stages are logged.
    """
    resolved_log_dir = project_root / "logs" / "gen_readme"
    return setup_readme_logger(
        log_dir=resolved_log_dir,
        script_name="gen_readme_cli",
        level=logging.INFO,
        console_output=True,
    )


def run_generation(args: argparse.Namespace, logger: logging.Logger) -> Path:
    """
    Execute README generation using parsed CLI arguments.

    Input:
        args (argparse.Namespace): Parsed CLI arguments.
        logger (logging.Logger): CLI audit logger.

    Output:
        Path: Generated README absolute path on success.

    Scenario:
        Main business call path for CLI mode.

    """
    project_root = get_project_root()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    template_path = project_root / "src" / "robocoin_dataset" / "readme" / "assets" / "readme.j2"
    output_path = dataset_path / "README.md"
    fixed_log_dir = project_root / "logs" / "gen_readme"

    logger.info(f"[CLI] dataset_path={dataset_path}")
    logger.info(f"[CLI] template_path={template_path}")
    logger.info(f"[CLI] output_path={output_path}")
    logger.info(f"[CLI] log_dir={fixed_log_dir}")

    generator = ReadmeGenerator(
        dataset_path=dataset_path,
        template_path=template_path,
        output_path=output_path,
        log_dir=fixed_log_dir,
    )
    return generator.generate_readme()


def run_batch_generation(args: argparse.Namespace, logger: logging.Logger) -> tuple[list[Path], list[str]]:
    """
    Execute README generation in batch mode from database-selected datasets.
    """
    db_cfg_path = Path(args.db_cfg_path).expanduser().resolve()
    if not db_cfg_path.exists():
        raise FileNotFoundError(f"[BATCH] Database config file not found: {db_cfg_path}")

    logger.info(f"[BATCH] db_cfg_path={db_cfg_path}")
    logger.info(f"[BATCH] ignore_uploaded={args.ignore_uploaded}")
    logger.info(f"[BATCH] target_dataset_uuid={args.target_dataset_uuid}")

    db = DatasetDatabase(db_cfg_path)
    with db.with_session() as session:
        tasks = list_readme_tasks(
            session=session,
            ignore_uploaded=args.ignore_uploaded,
            target_dataset_uuid=args.target_dataset_uuid,
        )

    if not tasks:
        logger.info("[BATCH] No eligible datasets found.")
        return [], []

    project_root = get_project_root()
    template_path = project_root / "src" / "robocoin_dataset" / "readme" / "assets" / "readme.j2"
    fixed_log_dir = project_root / "logs" / "gen_readme"
    results: list[Path] = []
    failed_dataset_uuids: list[str] = []

    logger.info(f"[BATCH] Found {len(tasks)} eligible datasets.")
    for dataset_uuid, convert_path in tasks:
        try:
            logger.info(
                "[BATCH] Generating README for dataset_uuid=%s, convert_path=%s",
                dataset_uuid,
                convert_path,
            )
            generator = ReadmeGenerator(
                dataset_path=convert_path,
                template_path=template_path,
                output_path=convert_path / "README.md",
                log_dir=fixed_log_dir,
            )
            results.append(generator.generate_readme())
            log_success(logger, f"[BATCH] README generated for dataset_uuid={dataset_uuid}")
        except Exception as exc:
            log_error(logger, f"[BATCH] Failed for dataset_uuid={dataset_uuid}: {exc}")
            logger.error("[BATCH] traceback follows:\n%s", traceback.format_exc())
            failed_dataset_uuids.append(dataset_uuid)
    return results, failed_dataset_uuids


def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI main function.

    Input:
        argv (Optional[list[str]]): Optional argument list for testability.

    Output:
        int: Process exit code.
            - 0: success
            - 1: runtime failure
            - 2: invalid CLI arguments (managed by argparse)

    Scenario:
        Script entry point for command-line execution.
    """
    try:
        args = parse_cli_args(argv)
        project_root = get_project_root()
        logger = create_cli_logger(project_root=project_root)
        if args.db_cfg_path:
            output_paths, failed_dataset_uuids = run_batch_generation(args=args, logger=logger)
            if output_paths:
                log_success(logger, f"[CLI] Batch README generation completed: {len(output_paths)} dataset(s)")
                for path in output_paths:
                    print(str(path))
            else:
                logger.info("[CLI] Batch mode finished with no generated README.")
            if failed_dataset_uuids:
                log_error(
                    logger,
                    "[CLI] Batch finished with failures. "
                    f"failed_count={len(failed_dataset_uuids)} "
                    f"failed_dataset_uuids={','.join(failed_dataset_uuids)}",
                )
                return 1
        else:
            if not args.dataset_path:
                raise ValueError("Either --dataset-path or --db-cfg-path must be provided.")
            output_path = run_generation(args=args, logger=logger)
            log_success(logger, f"[CLI] README generated successfully: {output_path}")
            print(str(output_path))
        return 0
    except SystemExit:
        # Keep argparse default behavior and exit code for invalid arguments/help.
        raise
    except Exception as exc:
        fallback_logger = logging.getLogger("gen_readme_cli_fallback")
        fallback_logger.setLevel(logging.ERROR)
        if not fallback_logger.handlers:
            fallback_logger.addHandler(logging.StreamHandler(sys.stderr))
        log_error(fallback_logger, f"[CLI] README generation failed: {exc}")
        fallback_logger.error("[CLI] traceback follows:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
