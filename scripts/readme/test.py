"""
README Generator Test Script

This script generates a sample README.md file from info.yaml template,
where all value fields are replaced with their key names as demonstration
values. This creates a template README that shows the structure and format.

Dependencies:
    - robocoin_dataset.readme.gen_readme: ReadmeGenerator class
    - robocoin_dataset.readme.utils: Utility functions for YAML loading
    - yaml: For YAML file operations
    - pathlib: For file path operations
    - sys: For exit codes
    - tempfile: For temporary file creation
    - logging: For audit logging

Usage:
    Run from command line:
        python scripts/readme/test.py

    Output:
        - Generated README: scripts/readme/sample.md
        - Log file: scripts/readme/logs/test_<timestamp>.log
        - Temporary YAML file (auto-deleted after use)
"""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from robocoin_dataset.readme.gen_readme import ReadmeGenerator
from robocoin_dataset.readme.utils import setup_readme_logger


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
            - info_yaml_path: Path to info.yaml
            - template_path: Path to readme.j2 template
            - output_path: Path for output sample.md
            - log_dir: Directory for log files

    Logic:
        1. Get project root directory
        2. Construct paths for:
           - Input YAML: src/robocoin_dataset/readme/assets/info.yaml
           - Template: src/robocoin_dataset/readme/assets/readme.j2
           - Output: scripts/readme/sample.md
           - Logs: scripts/readme/logs
        3. Return all paths as resolved absolute paths

    Usage:
        Called at script start to configure file paths.
    """
    project_root = get_project_root()

    # Input YAML file path (using info.yaml instead of info_sample.yaml)
    info_yaml_path = (
        project_root
        / "src"
        / "robocoin_dataset"
        / "readme"
        / "assets"
        / "info.yaml"
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
# YAML Value Replacement Functions
# ============================================================================


def get_demo_value_for_key(key: str) -> Any:
    """
    Get demonstration value for a specific key based on expected data type.

    Input:
        key (str): Field key name

    Output:
        Any: Demonstration value appropriate for the key type

    Logic:
        1. Check key name against known patterns
        2. For list-type fields: return list with key name as element
        3. For dict-type fields: return dict with example structure
        4. For string-type fields: return key name as string

    Usage:
        Called to create type-appropriate demonstration values for template rendering.

    Example:
        "task_categories" -> ["task_categories"]
        "extra_gated_fields" -> {"example_field": {"type": "text", "description": "example_field"}}
        "dataset_name" -> "dataset_name"
    """
    # Fields that should be lists (based on template usage)
    list_fields = {
        "task_categories",
        "language",
        "tags",
        "atomic_actions",
        "sub_tasks",
        "sensor_list",
        "came_info",
        "configs",
    }

    # Fields that should be dictionaries (based on template usage)
    dict_fields = {
        "extra_gated_fields",
        "statistics",
        "splits",
        "features",
        "authors",
        "objects",
        "scene_type",
    }

    if key in list_fields:
        # Return list with key name as demonstration element
        if key == "configs":
            # configs needs to be a list of dicts with config_name and data_files
            return [{"config_name": key, "data_files": key}]
        else:
            return [key]
    elif key in dict_fields:
        # Return dict with example structure
        if key == "extra_gated_fields":
            return {key: {"type": "text", "description": key}}
        elif key == "statistics":
            return {f"{key}_field": key}
        elif key == "splits":
            return {"train": key, "val": key, "test": key}
        elif key == "features":
            # Return dict with example observation features
            # Note: Template uses 'match' test which may not be available in Jinja2 environment
            # Return empty dict to avoid triggering the problematic code path
            # If match test is available, this will work; otherwise template will skip this section
            return {}
        elif key == "authors":
            return {"contributed_by": [key], "annotated_by": [key]}
        elif key == "objects":
            return [{"object_name": key, "level1": key}]
        elif key == "scene_type":
            return {"level1": key, "level2": key}
        else:
            return {key: key}
    else:
        # Default: return key name as string
        return key


def replace_values_with_keys(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """
    Recursively replace all value fields with their key names in YAML structure.

    Input:
        data (Dict[str, Any]): YAML data dictionary to process
        parent_key (str): Parent key name for nested structures (default: "")

    Output:
        Dict[str, Any]: Modified dictionary with values replaced by key names

    Logic:
        1. Iterate through all keys in the dictionary
        2. For each key-value pair:
           - If value is a dict with 'value' key: replace value['value'] with appropriate demo value
           - If value is a dict without 'value' key: recursively process nested dict
           - If value is a list: recursively process each item
           - Otherwise: keep original value
        3. Return modified dictionary

    Usage:
        Called to create demonstration template where all values are key names.
        This helps users understand the structure and expected field names.

    Example:
        Input: {"dataset_name": {"value": null}}
        Output: {"dataset_name": {"value": "dataset_name"}}
        Input: {"task_categories": {"value": null}}
        Output: {"task_categories": {"value": ["task_categories"]}}
    """
    result = {}
    for key, value in data.items():
        current_key = key if not parent_key else f"{parent_key}.{key}"

        if isinstance(value, dict):
            if "value" in value:
                # Replace value field with appropriate demonstration value
                modified_value = value.copy()
                demo_value = get_demo_value_for_key(key)
                modified_value["value"] = demo_value
                result[key] = modified_value
            else:
                # Recursively process nested dictionary
                result[key] = replace_values_with_keys(value, current_key)
        elif isinstance(value, list):
            # Process list items recursively
            result[key] = [
                replace_values_with_keys(item, current_key) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Keep original value for non-dict, non-list items
            result[key] = value

    return result


def create_demo_yaml_file(
    source_yaml_path: Path, logger: Any, temp_dir: Path
) -> Path:
    """
    Create a temporary YAML file with all values replaced by key names.

    Input:
        source_yaml_path (Path): Path to source info.yaml file
        logger (logging.Logger): Logger instance for error reporting
        temp_dir (Path): Directory for temporary file

    Output:
        Path: Path to created temporary YAML file

    Logic:
        1. Load source YAML file using utility function
        2. Replace all value fields with their key names
        3. Write modified data to temporary YAML file
        4. Return temporary file path

    Usage:
        Called to create demonstration YAML file before README generation.
        The temporary file will be used by ReadmeGenerator.

    Raises:
        FileNotFoundError: If source YAML file does not exist
        yaml.YAMLError: If YAML parsing or writing fails
        IOError: If file writing fails
    """
    logger.info(f"[DEMO_YAML] Loading source YAML: {source_yaml_path}")

    # Load original YAML (this will extract 'value' fields)
    # But we need the raw structure, so load it directly
    if not source_yaml_path.exists():
        error_msg = f"[DEMO_YAML] Source YAML file not found: {source_yaml_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        with open(source_yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if raw_data is None:
            logger.warning(f"[DEMO_YAML] Source YAML file is empty: {source_yaml_path}")
            raw_data = {}

        logger.info(f"[DEMO_YAML] Loaded {len(raw_data)} fields from source YAML")

        # Replace all values with key names
        logger.info("[DEMO_YAML] Replacing values with key names")
        demo_data = replace_values_with_keys(raw_data)

        # Create temporary YAML file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="demo_info_",
            dir=temp_dir,
            delete=False,
            encoding="utf-8",
        )
        temp_file_path = Path(temp_file.name)

        # Write modified YAML to temporary file
        yaml.safe_dump(
            demo_data,
            temp_file,
            default_flow_style=False,
            allow_unicode=False,
            sort_keys=False,
            width=120,
        )
        temp_file.close()

        logger.info(f"[DEMO_YAML] Created temporary demo YAML: {temp_file_path}")
        return temp_file_path

    except yaml.YAMLError as e:
        error_msg = f"[DEMO_YAML] Failed to parse or write YAML: {e}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"[DEMO_YAML] Unexpected error creating demo YAML: {e}"
        logger.error(error_msg)
        raise


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
        1. Setup logger for audit logging
        2. Setup all required file paths
        3. Validate input files exist
        4. Create temporary YAML file with values replaced by key names
        5. Create ReadmeGenerator instance with temporary YAML path
        6. Call generate_readme() to generate README
        7. Clean up temporary file
        8. Print success message with output path
        9. Return exit code

    Usage:
        Main entry point for the test script.
        Called when script is executed directly.

    Raises:
        FileNotFoundError: If input YAML or template file is missing
        Exception: Any other error during README generation
    """
    logger = None
    temp_yaml_path = None

    try:
        # Setup paths
        (
            project_root,
            info_yaml_path,
            template_path,
            output_path,
            log_dir,
        ) = setup_paths()

        # Setup logger
        logger = setup_readme_logger(
            log_dir=log_dir,
            script_name="test",
            level=logging.INFO,
            console_output=True,
        )
        logger.info("[MAIN] Starting README demonstration template generation")

        # Validate input files exist
        if not info_yaml_path.exists():
            error_msg = f"[MAIN] Input YAML file not found: {info_yaml_path}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            return 1

        if not template_path.exists():
            error_msg = f"[MAIN] Template file not found: {template_path}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            return 1

        # Create temporary directory for demo YAML file
        temp_dir = Path(tempfile.gettempdir())
        logger.info(f"[MAIN] Creating demonstration YAML file from: {info_yaml_path}")

        # Create demo YAML file with values replaced by key names
        temp_yaml_path = create_demo_yaml_file(
            source_yaml_path=info_yaml_path,
            logger=logger,
            temp_dir=temp_dir,
        )

        # Create ReadmeGenerator instance with temporary YAML path
        # Note: dataset_path is required but not used when custom paths are provided
        # Using project_root as a placeholder
        logger.info("[MAIN] Initializing ReadmeGenerator")
        generator = ReadmeGenerator(
            dataset_path=project_root,
            info_yaml_path=temp_yaml_path,
            template_path=template_path,
            output_path=output_path,
            log_dir=log_dir,
        )

        # Generate README
        logger.info("[MAIN] Generating README from demonstration template")
        generated_path = generator.generate_readme()

        # Clean up temporary YAML file
        if temp_yaml_path and temp_yaml_path.exists():
            try:
                temp_yaml_path.unlink()
                logger.info(f"[MAIN] Cleaned up temporary file: {temp_yaml_path}")
            except Exception as e:
                logger.warning(f"[MAIN] Failed to delete temporary file {temp_yaml_path}: {e}")

        # Print success message
        logger.info(f"[MAIN] README generation completed successfully: {generated_path}")
        print(f"[SUCCESS] Sample README generated: {generated_path}")
        print(f"[INFO] You can view the output at: {generated_path}")

        return 0

    except FileNotFoundError as e:
        error_msg = f"[MAIN] File not found: {e}"
        if logger:
            logger.error(error_msg)
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return 1
    except Exception as e:
        error_msg = f"[MAIN] README generation failed: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return 1
    finally:
        # Ensure temporary file is cleaned up even on error
        if temp_yaml_path and temp_yaml_path.exists():
            try:
                temp_yaml_path.unlink()
            except Exception:
                pass


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

