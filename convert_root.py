import os
import argparse
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.ROOTDataset import ROOTDataset

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--file_path", type=str, help="Path to the root file")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    root_dt = ROOTDataset.from_root_file(args.file_path, backend='pd')
    root_dt.save(args.save_path)
    print(root_dt.dataset.element_spec)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
