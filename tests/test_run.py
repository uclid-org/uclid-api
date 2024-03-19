import subprocess

from uclid import verify
from uclid.run import download_uclid

__author__ = "Federico Mora"
__copyright__ = "Federico Mora"
__license__ = "MIT"


def test_download():
    uclid = download_uclid()
    output = subprocess.run(
        [uclid, "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert output.returncode == 0
    output = output.stdout.decode("utf-8")
    assert "uclid 0.9.5" in output, f"uclid version not found in {output}"


def test_run():
    query = "module main { }"
    output = verify(query).split()
    expected = "Successfully parsed 1 and instantiated 1 module(s).".split()
    assert output == expected, f"Expected `{expected}` but got `{output}`"
