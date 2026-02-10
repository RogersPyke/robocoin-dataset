"""
README Generator Test Script

This script tests the ReadmeGenerator by generating a sample README.md file
from the sample YAML configuration file. It is used to observe the output
format of the generated README.

Dependencies:
    - robocoin_dataset.readme.gen_readme: ReadmeGenerator class
    - pathlib: For file path operations
    - sys: For exit codes

Usage:
    Run from command line:
        python scripts/readme/test.py

    Output:
        - Generated README: scripts/readme/sample.md
        - Log file: scripts/readme/logs/test_<timestamp>.log
"""

import sys
from pathlib import Path

from robocoin_dataset.readme.gen_readme import ReadmeGenerator


# ============================================================================
# Path Configuration
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory path.

    Input:
        None

    Output:
        Path: Absolute path to project root directory

    Logic:
        1. Get current script's directory
        2. Navigate up to project root (scripts/readme -> project root)
        3. Return resolved absolute path

    Usage:
        Used to construct absolute paths relative to project root.
    """
    script_dir = Path(__file__).parent.resolve()
    # Navigate from scripts/readme/ to project root
    project_root = script_dir.parent.parent
    return project_root


def setup_paths() -> tuple[Path, Path, Path, Path]:
    """
    Setup all required file paths for README generation.

    Input:
        None

    Output:
        tuple[Path, Path, Path, Path]: Tuple containing:
            - project_root: Project root directory
            - info_yaml_path: Path to info_sample.yaml
            - template_path: Path to readme.j2 template
            - output_path: Path for output sample.md
            - log_dir: Directory for log files

    Logic:
        1. Get project root directory
        2. Construct paths for:
           - Input YAML: src/robocoin_dataset/readme/assets/info_sample.yaml
           - Template: src/robocoin_dataset/readme/assets/readme.j2
           - Output: scripts/readme/sample.md
           - Logs: scripts/readme/logs
        3. Return all paths as resolved absolute paths

    Usage:
        Called at script start to configure file paths.
    """
    project_root = get_project_root()

    # Input YAML file path
    info_yaml_path = (
        project_root
        / "src"
        / "robocoin_dataset"
        / "readme"
        / "assets"
        / "info_sample.yaml"
    )

    # Template file path
    template_path = (
        project_root
        / "src"
        / "robocoin_dataset"
        / "readme"
        / "assets"
        / "readme.j2"
    )

    # Output file path (in scripts/readme directory)
    script_dir = Path(__file__).parent.resolve()
    output_path = script_dir / "sample.md"

    # Log directory path (in scripts/readme/logs)
    log_dir = script_dir / "logs"

    return project_root, info_yaml_path, template_path, output_path, log_dir


# ============================================================================
# Main Execution
# ============================================================================


def main() -> int:
    """
    Main function to generate sample README from YAML configuration.

    Input:
        None (uses command line environment)

    Output:
        int: Exit code (0 for success, 1 for failure)

    Logic:
        1. Setup all required file paths
        2. Validate input files exist
        3. Create ReadmeGenerator instance with custom paths
        4. Call generate_readme() to generate README
        5. Print success message with output path
        6. Return exit code

    Usage:
        Main entry point for the test script.
        Called when script is executed directly.

    Raises:
        FileNotFoundError: If input YAML or template file is missing
        Exception: Any other error during README generation
    """
    try:
        # Setup paths
        (
            project_root,
            info_yaml_path,
            template_path,
            output_path,
            log_dir,
        ) = setup_paths()

        # Validate input files exist
        if not info_yaml_path.exists():
            print(
                f"[ERROR] Input YAML file not found: {info_yaml_path}",
                file=sys.stderr,
            )
            return 1

        if not template_path.exists():
            print(
                f"[ERROR] Template file not found: {template_path}",
                file=sys.stderr,
            )
            return 1

        # Create ReadmeGenerator instance
        # Note: dataset_path is required but not used when custom paths are provided
        # Using project_root as a placeholder
        generator = ReadmeGenerator(
            dataset_path=project_root,
            info_yaml_path=info_yaml_path,
            template_path=template_path,
            output_path=output_path,
            log_dir=log_dir,
        )

        # Generate README
        generated_path = generator.generate_readme()

        # Print success message
        print(f"[SUCCESS] Sample README generated: {generated_path}")
        print(f"[INFO] You can view the output at: {generated_path}")

        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] README generation failed: {e}", file=sys.stderr)
        return 1


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

