"""Exercise archive integrity and the boundary around existing filesystem paths."""

import hashlib
from pathlib import Path
import stat
import zipfile

import pytest

import download_data
from setup_paths import ensure_alias


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setattr(download_data, "DATA_ROOT", root)
    monkeypatch.setattr(download_data, "MANIFEST", tmp_path / "work/manifest.json")
    return root


def test_extract_and_detect_same_size_corruption(data_root):
    archive = data_root / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("train/sample.zarr/0/zarr.json", b'{"shape":[2,3,4,5]}')
        bundle.writestr("sample_submission.csv", b"id,row_type\n0,node\n")
    expected = hashlib.md5(archive.read_bytes()).hexdigest()
    manifest = download_data.extract(archive, expected)
    assert manifest["archive_md5"] == expected
    assert len(download_data.verify(download_data.MANIFEST, checksums=True)["files"]) == 2
    sample = data_root / "sample_submission.csv"
    sample.write_bytes(sample.read_bytes().replace(b"node", b"edge"))
    with pytest.raises(RuntimeError, match="CRC mismatch"):
        download_data.verify(download_data.MANIFEST, checksums=True)


@pytest.mark.parametrize("name,is_symlink", [("../escape", False), ("/escape", False), ("link", True)])
def test_unsafe_archive_does_not_extract(data_root, name, is_symlink):
    archive = data_root / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        member = zipfile.ZipInfo(name)
        if is_symlink:
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(member, "outside")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        download_data.extract(archive)
    assert not download_data.MANIFEST.exists()
    assert list(data_root.iterdir()) == [archive]


def test_archive_checksum_failure_leaves_data_untouched(data_root):
    archive = data_root / "inputs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sample_submission.csv", "example")
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        download_data.extract(archive, "0" * 32)
    assert not (data_root / "sample_submission.csv").exists()
    assert not download_data.MANIFEST.exists()


def test_alias_is_idempotent_and_preserves_existing_paths(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    assert ensure_alias(alias, target) == "created"
    assert ensure_alias(alias, target) == "already-correct"
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "user-file").write_text("keep")
    with pytest.raises(RuntimeError, match="real path"):
        ensure_alias(occupied, target)
    assert (occupied / "user-file").read_text() == "keep"
    wrong = tmp_path / "wrong"
    wrong.symlink_to(occupied)
    with pytest.raises(RuntimeError, match="elsewhere"):
        ensure_alias(wrong, target)
    assert wrong.resolve() == occupied.resolve()
