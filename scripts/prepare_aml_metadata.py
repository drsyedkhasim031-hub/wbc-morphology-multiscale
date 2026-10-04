"""Convert TCIA annotations.dat into manifest metadata without inventing patients."""

import argparse
import pandas as pd
from wbc.labels import AML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = pd.read_csv(
        args.annotations,
        sep=r"\s+",
        names=["path", "original_label", "reannotation_1", "reannotation_2"],
        keep_default_na=False,
    )
    if not df.original_label.isin(AML).all():
        raise ValueError("Unexpected original AML label")
    df["collection"], df["dataset_version"] = "full", "TCIA Version 1 2019-10-24"
    df["patient_id"], df["slide_id"] = "", ""
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
