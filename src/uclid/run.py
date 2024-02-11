import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
import zipfile

import wget

__author__ = "Federico Mora"
__copyright__ = "Federico Mora"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

LATEST_RELEASE_URL = "https://github.com/uclid-org/uclid/releases/download/v0.9.5d-prerelease/uclid-0.9.5.zip"  # noqa: E501


def verify(query: str):
    uclid = download_uclid()
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(query.encode())
        tmp.flush()
        _logger.info(f"Running {uclid} on {tmp.name}")
        output = subprocess.run([uclid, tmp.name], capture_output=True)
        return output.stdout.decode("utf-8")


def download_uclid():
    # Get the directory of this file
    base = pathlib.Path(__file__).parent.resolve()
    _logger.info(f"Base directory: {base}")

    # Get the other directories we'll use
    build = os.path.join(base, "build")
    uclid = os.path.join(build, "uclid-0.9.5", "bin", "uclid")
    # if on windows, add .bat to uclid
    if os.name == "nt":
        uclid += ".bat"
    _logger.info(f"Uclid executable: {uclid}")

    # if uclid exists, return it
    if os.path.exists(uclid):
        _logger.info("Uclid exists")
        return uclid

    # delete build directory if it exists
    _logger.info("Deleting build directory if it exists")
    if os.path.exists(build):
        shutil.rmtree(build)
        _logger.info("Build directory deleted")

    # make a build directory if it doesn't exist
    _logger.info("Making build directory")
    os.makedirs(build)

    # download uclid
    _logger.info("Downloading Uclid")
    wget.download(LATEST_RELEASE_URL, os.path.join(build, "uclid.zip"))
    _logger.info("Uclid downloaded")

    # unzip uclid.zip
    _logger.info("Unzipping Uclid")
    with zipfile.ZipFile(os.path.join(build, "uclid.zip"), "r") as zip_ref:
        zip_ref.extractall(build)
    _logger.info("Uclid unzipped")

    # make uclid executable
    _logger.info("Making Uclid executable")
    os.chmod(uclid, 0o755)
    _logger.info("Uclid is now executable")

    return uclid
