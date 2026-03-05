"""
README CLI entry script.

Purpose:
    Provide a command-line interface for generating dataset README files
    via robocoin_dataset.readme.gen_readme.ReadmeGenerator.

Dependencies:
    - argparse: Parse command-line arguments
    - pathlib: Path normalization
    - traceback: Error stack capture
    - robocoin_dataset.readme.gen_readme: Core README generator class
    - robocoin_dataset.readme.utils: Logger setup utility
    - robocoin_dataset.utils.log_config: Colored success/error logs

Usage examples:
    1) Use default metadata path under dataset directory:
       python scripts/readme/gen_readme.py --dataset-path /data/my_dataset

    2) Use custom metadata path only:
       python scripts/readme/gen_readme.py \
           --dataset-path ~/projects/TestDatasets_0/Agilex_Cobot_Magic_pour_water_into_cup_0_qced_hardlink \
           --info-yaml-path ~/projects/TestDatasets_0/local_dataset_info.yaml

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

from robocoin_dataset.readme.gen_readme import ReadmeGenerator
from robocoin_dataset.readme.utils import setup_readme_logger
from robocoin_dataset.utils.log_config import log_error, log_success


def parse_cli_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments for README generation.

    Input:
        argv (Optional[list[str]]): Argument list. If None, argparse reads sys.argv.

    Output:
        argparse.Namespace: Parsed arguments object with fields:
            - dataset_path (str, required)
            - local_dataset_info_path (Optional[str])

    Scenario:
        Called by main() as the only CLI input entry.

    Note:
        The system uses priority-based YAML file location:
        1. First searches for local_dataset_info.yaml in dataset_path (first level)
        2. If not found, validates custom path from --local-dataset-info-path
        3. If neither found, reports detailed error and exits
    """
    parser = argparse.ArgumentParser(
        prog="gen_readme.py",
        description="Generate dataset README.md from YAML metadata and Jinja2 template.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Dataset/hardlink directory path (root directory where data is stored).",
    )
    parser.add_argument(
        "--local-dataset-info-path",
        default=None,
        help="Optional custom path to local_dataset_info.yaml. "
        "If omitted, system searches for 'local_dataset_info.yaml' at dataset_path first level. "
        "If neither found, error is reported and process exits.",
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

    Note:
        The ReadmeGenerator will automatically locate the YAML file using:
        1. Search in dataset_path first level for local_dataset_info.yaml
        2. Validate custom path if provided via --local-dataset-info-path
        3. Raise error if no valid file found
    """
    project_root = get_project_root()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    template_path = project_root / "src" / "robocoin_dataset" / "readme" / "assets" / "readme.j2"
    output_path = dataset_path / "README.md"
    fixed_log_dir = project_root / "logs" / "gen_readme"

    logger.info(f"[CLI] dataset_path={dataset_path}")
    logger.info(f"[CLI] local_dataset_info_path={args.local_dataset_info_path}")
    logger.info(f"[CLI] template_path={template_path}")
    logger.info(f"[CLI] output_path={output_path}")
    logger.info(f"[CLI] log_dir={fixed_log_dir}")

    generator = ReadmeGenerator(
        dataset_path=dataset_path,
        info_yaml_path=args.local_dataset_info_path,
        template_path=template_path,
        output_path=output_path,
        log_dir=fixed_log_dir,
    )
    return generator.generate_readme()


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
