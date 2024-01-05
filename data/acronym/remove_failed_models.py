import shutil
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove failed model files"
    )

    parser.add_argument(
        "failed_model_file",
        type=str,
        help="Failed model log file",
    )

    args = parser.parse_args()

    failed_model_file = Path(args.failed_model_file)
    assert failed_model_file.is_file(), f"{failed_model_file=}"

    with failed_model_file.open("r") as f:
        lines = f.readlines()

    for line in lines:
        model_dir = Path(line.strip())
        assert model_dir.is_dir()
        shutil.rmtree(model_dir)

    print(f"Removed {len(lines)} failed model directories")
