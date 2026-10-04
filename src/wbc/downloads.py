"""Dataset access metadata and verified download utility (no embedded credentials)."""

from pathlib import Path
import hashlib
import requests
from .utils import write_json

SOURCES = {
    "raabin": {
        "url": "https://raabindata.com/",
        "selected_native": 14514,
        "common_five": 14514,
        "instructions": "Download training and Test-A from the original release. Exclude Test-B. Preserve collection names. Review current source terms.",
    },
    "pbc": {
        "url": "https://data.mendeley.com/datasets/snkd93bnjr/1",
        "doi": "10.17632/snkd93bnjr.1",
        "released": 17092,
        "selected_native": 13193,
        "common_five": 10298,
        "license": "CC BY 4.0",
        "instructions": "Download original JPG archive. Exclude erythroblasts and platelets for native WBC task. Exclude immature granulocytes additionally for common-five.",
    },
    "aml": {
        "url": "https://www.cancerimagingarchive.net/collection/aml-cytomorphology_lmu/",
        "doi": "10.7937/tcia.2019.36f5o9ld",
        "selected_native": 18365,
        "common_five": 14833,
        "license": "CC BY 3.0",
        "instructions": "Use TCIA's image download (approximately 11 GB, Aspera) plus abbreviations and annotation files. Do not infer patient IDs from class-number filenames.",
    },
    "bloodmnist_demo": {
        "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
        "md5": "7053d0359d879ad8a5505303e11de1dc",
        "license": "CC BY 4.0",
        "instructions": "28x28 PBC-derived dataset; only for an explicitly separate demonstration. Its partitions and images do not implement the manuscript protocol.",
    },
}


def download_file(url, destination, expected_hash=None, algorithm="sha256"):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.new(algorithm)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("Received HTML instead of a dataset; use the provider download page")
        with temporary.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    if expected_hash and digest.hexdigest() != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("Download checksum mismatch")
    temporary.replace(destination)
    write_json(
        str(destination) + ".source.json",
        {"source_url": url, "algorithm": algorithm, "checksum": digest.hexdigest()},
    )
    return destination
