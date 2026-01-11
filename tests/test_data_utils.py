"""
Unit tests for app/src/data_utils/commons.py

Tests cover:
- extract_dataset(): ZIP file extraction
- list_directory_structure(): Directory structure listing
- download_from_kaggle(): Kaggle API integration
- download_and_extract(): Orchestration workflow
"""
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.src.data_utils.commons import (
    extract_dataset,
    list_directory_structure,
    download_from_kaggle,
    download_and_extract
)


class TestExtractDataset:
    """Test suite for extract_dataset function"""

    def test_extract_valid_zip_success(self, sample_zip, tmp_path):
        """Test successful extraction of a valid ZIP file"""
        extract_to = tmp_path / "extracted"

        extract_dataset(str(sample_zip), str(extract_to))

        # Verify extraction directory was created
        assert extract_to.exists()

        # Verify files were extracted
        assert (extract_to / "test.txt").exists()
        assert (extract_to / "folder" / "image1.jpg").exists()
        assert (extract_to / "folder" / "image2.jpg").exists()

    def test_extract_creates_directory_if_not_exists(self, sample_zip, tmp_path):
        """Test that extraction creates the target directory if it doesn't exist"""
        extract_to = tmp_path / "new_directory"

        # Ensure directory doesn't exist before extraction
        assert not extract_to.exists()

        extract_dataset(str(sample_zip), str(extract_to))

        # Verify directory was created
        assert extract_to.exists()

    def test_extract_missing_zip_file(self, tmp_path, caplog):
        """Test error handling when ZIP file doesn't exist"""
        non_existent_zip = tmp_path / "missing.zip"
        extract_to = tmp_path / "extracted"

        extract_dataset(str(non_existent_zip), str(extract_to))

        # Verify error was logged
        assert "Arquivo não encontrado" in caplog.text

        # Verify extraction directory was not created
        assert not extract_to.exists()

    def test_extract_corrupted_zip_file(self, corrupted_zip, tmp_path, caplog):
        """Test error handling when ZIP file is corrupted"""
        extract_to = tmp_path / "extracted"

        extract_dataset(str(corrupted_zip), str(extract_to))

        # Verify error was logged
        assert "corrompido" in caplog.text or "não é um ZIP válido" in caplog.text

    def test_extract_existing_directory_not_overwritten(self, sample_zip, tmp_path):
        """Test that extraction works even if directory already exists"""
        extract_to = tmp_path / "existing"
        extract_to.mkdir()

        # Create a marker file to verify it's not deleted
        marker_file = extract_to / "marker.txt"
        marker_file.write_text("existing content")

        extract_dataset(str(sample_zip), str(extract_to))

        # Verify original file still exists
        assert marker_file.exists()
        assert marker_file.read_text() == "existing content"

        # Verify new files were also extracted
        assert (extract_to / "test.txt").exists()


class TestListDirectoryStructure:
    """Test suite for list_directory_structure function"""

    def test_list_valid_directory(self, sample_directory, capsys):
        """Test listing a valid directory structure"""
        list_directory_structure(str(sample_directory))

        captured = capsys.readouterr()

        # Verify output contains expected elements
        assert "Estrutura de:" in captured.out
        assert "subdir1" in captured.out
        assert "subdir2" in captured.out
        assert "file0.txt" in captured.out

    def test_list_shows_first_five_files_only(self, sample_directory, capsys):
        """Test that only first 5 files are shown in a directory"""
        list_directory_structure(str(sample_directory))

        captured = capsys.readouterr()

        # We created 7 files, should show first 5 and a message about others
        assert "file0.txt" in captured.out
        assert "file4.txt" in captured.out
        assert "outros arquivos" in captured.out or "..." in captured.out

    def test_list_non_existent_directory(self, tmp_path, caplog):
        """Test warning when directory doesn't exist"""
        non_existent = tmp_path / "does_not_exist"

        list_directory_structure(str(non_existent))

        # Verify warning was logged
        assert "Caminho não existe" in caplog.text

    def test_list_empty_directory(self, tmp_path, capsys):
        """Test listing an empty directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        list_directory_structure(str(empty_dir))

        captured = capsys.readouterr()

        # Should show the directory structure
        assert "Estrutura de:" in captured.out


class TestDownloadFromKaggle:
    """Test suite for download_from_kaggle function"""

    def test_download_import_error(self, tmp_path, caplog):
        """Test handling when kaggle package is not available"""
        output_path = tmp_path / "downloads"

        # Mock the import statement inside the function
        with patch('builtins.__import__', side_effect=ImportError("No kaggle module")):
            result = download_from_kaggle("test/dataset", str(output_path))

        # Verify function returns None on ImportError
        assert result is None

        # Verify error was logged
        assert "Erro ao conectar com o Kaggle" in caplog.text

    def test_download_authentication_error(self, tmp_path, caplog, mocker):
        """Test handling when Kaggle authentication fails"""
        output_path = tmp_path / "downloads"

        # Mock at the module import level
        mock_api_instance = MagicMock()
        mock_api_instance.authenticate.side_effect = OSError("Credentials not found")

        mock_kaggle_module = MagicMock()
        mock_kaggle_module.KaggleApi.return_value = mock_api_instance

        mocker.patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': mock_kaggle_module})

        result = download_from_kaggle("test/dataset", str(output_path))

        # Verify function returns None on authentication error
        assert result is None

        # Verify error was logged
        assert "Erro ao conectar com o Kaggle" in caplog.text

    def test_download_success(self, tmp_path, mocker):
        """Test successful download from Kaggle"""
        output_path = tmp_path / "downloads"
        dataset_slug = "test/dataset"

        # Create a fake ZIP file that would be "downloaded"
        output_path.mkdir()
        zip_path = output_path / "dataset.zip"

        # Mock the download to actually create the file
        def mock_download(*args, **kwargs):
            zip_path.write_bytes(b"fake zip content")

        mock_api_instance = MagicMock()
        mock_api_instance.dataset_download_files = mock_download

        mock_kaggle_module = MagicMock()
        mock_kaggle_module.KaggleApi.return_value = mock_api_instance

        mocker.patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': mock_kaggle_module})

        result = download_from_kaggle(dataset_slug, str(output_path))

        # Verify API methods were called
        mock_api_instance.authenticate.assert_called_once()

        # Verify return value contains expected path
        assert result is not None
        assert "dataset.zip" in result

    def test_download_creates_output_directory(self, tmp_path, mocker):
        """Test that download creates output directory if it doesn't exist"""
        output_path = tmp_path / "new_downloads"

        mock_api_instance = MagicMock()

        mock_kaggle_module = MagicMock()
        mock_kaggle_module.KaggleApi.return_value = mock_api_instance

        mocker.patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': mock_kaggle_module})

        # Simulate successful download
        download_from_kaggle("test/dataset", str(output_path))

        # Verify directory was created
        assert output_path.exists()

    def test_download_api_exception(self, tmp_path, caplog, mocker):
        """Test handling when Kaggle API raises an exception during download"""
        output_path = tmp_path / "downloads"

        mock_api_instance = MagicMock()
        mock_api_instance.dataset_download_files.side_effect = Exception("Download failed")

        mock_kaggle_module = MagicMock()
        mock_kaggle_module.KaggleApi.return_value = mock_api_instance

        mocker.patch.dict('sys.modules', {'kaggle.api.kaggle_api_extended': mock_kaggle_module})

        result = download_from_kaggle("test/dataset", str(output_path))

        # Verify function returns None on download error
        assert result is None

        # Verify error was logged
        assert "Erro durante download" in caplog.text


class TestDownloadAndExtract:
    """Test suite for download_and_extract orchestration function"""

    def test_orchestration_happy_path(self, tmp_path, mocker):
        """Test successful orchestration of download and extraction"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dataset_slug = "test/dataset"

        # DON'T create ZIP file - we want download_from_kaggle to be called
        zip_path = data_dir / "dataset.zip"

        # Mock both functions
        mock_download = mocker.patch(
            'app.src.data_utils.commons.download_from_kaggle',
            return_value=str(zip_path)
        )
        mock_extract = mocker.patch('app.src.data_utils.commons.extract_dataset')

        # Don't create extract_path yet, so extract_dataset will be called
        result = download_and_extract(dataset_slug, str(data_dir))

        # Verify functions were called
        mock_download.assert_called_once()
        mock_extract.assert_called_once()

        # Verify return value
        assert result is not None
        assert "dataset" in str(result)

    def test_skip_download_if_zip_exists(self, tmp_path, mocker):
        """Test that download is skipped if ZIP file already exists"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dataset_slug = "test/dataset"

        # Create existing ZIP file
        zip_path = data_dir / "dataset.zip"
        zip_path.write_bytes(b"existing zip")

        mock_download = mocker.patch('app.src.data_utils.commons.download_from_kaggle')
        mock_extract = mocker.patch('app.src.data_utils.commons.extract_dataset')

        # Create extraction path to skip extraction too
        extract_path = data_dir / "dataset"
        extract_path.mkdir()

        result = download_and_extract(dataset_slug, str(data_dir))

        # Verify download was NOT called
        mock_download.assert_not_called()

        # Verify result is the extraction path
        assert result == str(extract_path)

    def test_skip_extraction_if_data_exists(self, tmp_path, mocker):
        """Test that extraction is skipped if data directory already exists"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dataset_slug = "test/dataset"

        # Create existing ZIP and extracted directory
        zip_path = data_dir / "dataset.zip"
        zip_path.write_bytes(b"existing zip")

        extract_path = data_dir / "dataset"
        extract_path.mkdir()

        mock_download = mocker.patch('app.src.data_utils.commons.download_from_kaggle')
        mock_extract = mocker.patch('app.src.data_utils.commons.extract_dataset')

        result = download_and_extract(dataset_slug, str(data_dir))

        # Verify neither download nor extract were called
        mock_download.assert_not_called()
        mock_extract.assert_not_called()

        # Verify result is correct
        assert result == str(extract_path)

    def test_return_none_on_download_failure(self, tmp_path, mocker):
        """Test that function returns None when download fails"""
        data_dir = tmp_path / "data"
        dataset_slug = "test/dataset"

        # Mock download to fail
        mock_download = mocker.patch(
            'app.src.data_utils.commons.download_from_kaggle',
            return_value=None
        )

        result = download_and_extract(dataset_slug, str(data_dir))

        # Verify function returns None
        assert result is None

        # Verify download was attempted
        mock_download.assert_called_once()

    def test_proper_path_construction(self, tmp_path, mocker):
        """Test that paths are constructed correctly from dataset slug"""
        data_dir = tmp_path / "custom_data"
        dataset_slug = "owner/my-dataset-name"

        # Mock download and extract
        zip_path = data_dir / "my-dataset-name.zip"
        extract_path = data_dir / "my-dataset-name"

        mocker.patch(
            'app.src.data_utils.commons.download_from_kaggle',
            return_value=str(zip_path)
        )
        mocker.patch('app.src.data_utils.commons.extract_dataset')

        extract_path.mkdir(parents=True)

        result = download_and_extract(dataset_slug, str(data_dir))

        # Verify path contains correct dataset name
        assert "my-dataset-name" in str(result)
