import os
import argparse
#

from src.data.ROOTDataset import ROOTDataset

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--file_path", type=str, help="Path to the root file")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    root_dt = ROOTDataset.from_root_file(args.file_path)
    root_dt.save(args.save_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
