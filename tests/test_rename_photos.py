"""Unit tests for rename_photos module."""

import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from rename_photos_ai.rename_photos import (
    encode_image_from_bytes,
    identify_movie,
    preprocess_image,
    process_photos,
    sanitize_filename,
)


@pytest.fixture
def mock_anthropic_client(mocker):
    """Create a mock Anthropic client with autospec."""
    mock_client = mocker.MagicMock()
    mock_client.messages = mocker.MagicMock()
    mock_client.messages.create = mocker.MagicMock()
    return mock_client


@pytest.fixture
def mock_image(mocker):
    """Create a mock PIL Image with autospec."""
    return mocker.MagicMock(spec=Image.Image)


@pytest.fixture
def mock_path(mocker):
    """Create a mock Path with autospec."""
    return mocker.MagicMock(spec=Path)


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            pytest.param(
                "john wick",
                "John_Wick",
                id="simple_title_with_spaces"
            ),
            pytest.param(
                "The Dark Knight",
                "The_Dark_Knight",
                id="title_with_multiple_words"
            ),
            pytest.param(
                "Spider-Man: Into the Spider-Verse",
                "Spider-man_Into_The_Spider-verse",
                id="title_with_colon_and_hyphen"
            ),
            pytest.param(
                "The Matrix",
                "The_Matrix",
                id="already_capitalized"
            ),
            pytest.param(
                "movie/with\\invalid*chars?",
                "Moviewithinvalidchars",
                id="title_with_invalid_filesystem_chars"
            ),
            pytest.param(
                "movie  with   multiple    spaces",
                "Movie_With_Multiple_Spaces",
                id="title_with_multiple_spaces"
            ),
            pytest.param(
                "_leading_and_trailing_",
                "Leading_And_Trailing",
                id="leading_and_trailing_underscores"
            ),
            pytest.param(
                "UPPERCASE MOVIE",
                "Uppercase_Movie",
                id="all_uppercase"
            ),
        ],
    )
    def test_sanitize_filename(self, title, expected):
        """Test sanitize_filename with various inputs."""
        result = sanitize_filename(title)
        assert result == expected


class TestEncodeImageFromBytes:
    """Tests for encode_image_from_bytes function."""

    @pytest.mark.parametrize(
        "image_data,expected_type",
        [
            pytest.param(
                b"fake_image_data",
                str,
                id="basic_image_data"
            ),
            pytest.param(
                b"",
                str,
                id="empty_image_data"
            ),
        ],
    )
    def test_encode_image_from_bytes(self, image_data, expected_type):
        """Test encode_image_from_bytes returns base64 string."""
        image_bytes = BytesIO(image_data)
        result = encode_image_from_bytes(image_bytes)

        assert isinstance(result, expected_type)
        assert result == base64.standard_b64encode(image_data).decode('utf-8')


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    def test_preprocess_image_rgb_conversion(self, mocker, tmp_path):
        """Test that PNG with transparency is converted to RGB."""
        mock_open = mocker.patch('rename_photos_ai.rename_photos.Image.open')
        mock_img = mocker.MagicMock(spec=Image.Image)
        mock_img.mode = 'RGBA'
        mock_img.size = (1000, 1000)
        mock_img.split.return_value = [None, None, None, mocker.MagicMock()]
        mock_open.return_value = mock_img

        mock_background = mocker.MagicMock(spec=Image.Image)
        mock_background.size = (1000, 1000)
        mock_background.save = mocker.MagicMock()
        mock_new = mocker.patch('rename_photos_ai.rename_photos.Image.new', return_value=mock_background)

        test_path = tmp_path / "test.png"
        test_path.touch()

        result = preprocess_image(test_path)

        assert isinstance(result, BytesIO)
        mock_open.assert_called_once_with(test_path)
        mock_new.assert_called_once_with('RGB', (1000, 1000), (255, 255, 255))
        mock_background.paste.assert_called_once()
        mock_background.save.assert_called_once()

    def test_preprocess_image_resize_large_image(self, mocker, tmp_path):
        """Test that large images are resized to 2048px max dimension."""
        mock_open = mocker.patch('rename_photos_ai.rename_photos.Image.open')
        mock_img = mocker.MagicMock(spec=Image.Image)
        mock_img.mode = 'RGB'
        mock_img.size = (4000, 3000)
        mock_resized = mocker.MagicMock(spec=Image.Image)
        mock_img.resize.return_value = mock_resized
        mock_open.return_value = mock_img

        test_path = tmp_path / "test.jpg"
        test_path.touch()

        result = preprocess_image(test_path)

        assert isinstance(result, BytesIO)
        mock_img.resize.assert_called_once()
        # Check that resize was called with correct dimensions (2048, 1536)
        called_size = mock_img.resize.call_args[0][0]
        assert called_size[0] == 2048
        assert called_size[1] == 1536

    def test_preprocess_image_no_resize_small_image(self, mocker, tmp_path):
        """Test that small images are not resized."""
        mock_open = mocker.patch('rename_photos_ai.rename_photos.Image.open')
        mock_img = mocker.MagicMock(spec=Image.Image)
        mock_img.mode = 'RGB'
        mock_img.size = (1000, 800)
        mock_open.return_value = mock_img

        test_path = tmp_path / "test.jpg"
        test_path.touch()

        result = preprocess_image(test_path)

        assert isinstance(result, BytesIO)
        mock_img.resize.assert_not_called()
        mock_img.save.assert_called_once()


class TestIdentifyMovie:
    """Tests for identify_movie function."""

    def test_identify_movie_success(self, mocker, mock_anthropic_client, tmp_path):
        """Test successful movie identification."""
        # Mock preprocess_image
        mock_preprocess = mocker.patch('rename_photos_ai.rename_photos.preprocess_image')
        mock_bytes = BytesIO(b'fake_preprocessed_data')
        mock_preprocess.return_value = mock_bytes

        # Mock encode_image_from_bytes
        mock_encode = mocker.patch('rename_photos_ai.rename_photos.encode_image_from_bytes')
        mock_encode.return_value = 'base64_encoded_string'

        # Mock API response
        mock_response = mocker.MagicMock()
        mock_content = mocker.MagicMock()
        mock_content.text = "The Matrix"
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        test_path = tmp_path / "test.jpg"
        test_path.touch()

        result = identify_movie(mock_anthropic_client, test_path)

        assert result == "The Matrix"
        mock_preprocess.assert_called_once_with(test_path)
        mock_encode.assert_called_once_with(mock_bytes)
        mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.parametrize(
        "api_response,expected_title",
        [
            pytest.param(
                "Inception",
                "Inception",
                id="simple_title"
            ),
            pytest.param(
                "  The Dark Knight  ",
                "The Dark Knight",
                id="title_with_whitespace"
            ),
            pytest.param(
                "Unknown",
                "Unknown",
                id="unknown_movie"
            ),
        ],
    )
    def test_identify_movie_various_responses(
        self, mocker, mock_anthropic_client, tmp_path, api_response, expected_title
    ):
        """Test identify_movie with various API responses."""
        mock_preprocess = mocker.patch('rename_photos_ai.rename_photos.preprocess_image')
        mock_preprocess.return_value = BytesIO(b'fake_data')

        mock_encode = mocker.patch('rename_photos_ai.rename_photos.encode_image_from_bytes')
        mock_encode.return_value = 'encoded'

        mock_response = mocker.MagicMock()
        mock_content = mocker.MagicMock()
        mock_content.text = api_response
        mock_response.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_response

        test_path = tmp_path / "test.jpg"
        test_path.touch()

        result = identify_movie(mock_anthropic_client, test_path)

        assert result == expected_title


class TestProcessPhotos:
    """Tests for process_photos function."""

    def test_process_photos_success(self, mocker, mock_anthropic_client, tmp_path):
        """Test successful processing of photos."""
        # Create test directories
        process_dir = tmp_path / "process"
        renamed_dir = tmp_path / "renamed"
        original_dir = tmp_path / "original"
        process_dir.mkdir()
        renamed_dir.mkdir()
        original_dir.mkdir()

        # Create test image file
        test_image = process_dir / "IMG_001.jpg"
        test_image.write_text("fake image data")

        # Mock identify_movie
        mock_identify = mocker.patch('rename_photos_ai.rename_photos.identify_movie')
        mock_identify.return_value = "The Matrix"

        # Mock Anthropic client creation
        mock_anthropic = mocker.patch('rename_photos_ai.rename_photos.Anthropic')
        mock_anthropic.return_value = mock_anthropic_client

        process_photos(process_dir, renamed_dir, original_dir, "fake_api_key")

        # Check that identify_movie was called
        mock_identify.assert_called_once()

        # Check that file was moved to renamed directory
        renamed_file = renamed_dir / "The_Matrix.jpg"
        assert renamed_file.exists()

        # Check that original was copied
        original_file = original_dir / "The_Matrix.jpg"
        assert original_file.exists()

    def test_process_photos_no_images(self, mocker, mock_anthropic_client, tmp_path, capsys):
        """Test processing when no images are found."""
        process_dir = tmp_path / "process"
        renamed_dir = tmp_path / "renamed"
        original_dir = tmp_path / "original"
        process_dir.mkdir()
        renamed_dir.mkdir()
        original_dir.mkdir()

        mock_anthropic = mocker.patch('rename_photos_ai.rename_photos.Anthropic')
        mock_anthropic.return_value = mock_anthropic_client

        process_photos(process_dir, renamed_dir, original_dir, "fake_api_key")

        captured = capsys.readouterr()
        assert "No image files found" in captured.out

    def test_process_photos_duplicate_handling(self, mocker, mock_anthropic_client, tmp_path):
        """Test that duplicate filenames are handled correctly."""
        process_dir = tmp_path / "process"
        renamed_dir = tmp_path / "renamed"
        original_dir = tmp_path / "original"
        process_dir.mkdir()
        renamed_dir.mkdir()
        original_dir.mkdir()

        # Create existing file with same name
        existing_file = renamed_dir / "The_Matrix.jpg"
        existing_file.write_text("existing")

        # Create test image files
        test_image1 = process_dir / "IMG_001.jpg"
        test_image1.write_text("fake image data 1")

        mock_identify = mocker.patch('rename_photos_ai.rename_photos.identify_movie')
        mock_identify.return_value = "The Matrix"

        mock_anthropic = mocker.patch('rename_photos_ai.rename_photos.Anthropic')
        mock_anthropic.return_value = mock_anthropic_client

        process_photos(process_dir, renamed_dir, original_dir, "fake_api_key")

        # Check that duplicate was renamed with _1 suffix
        duplicate_file = renamed_dir / "The_Matrix_1.jpg"
        assert duplicate_file.exists()
        assert existing_file.exists()

    @pytest.mark.parametrize(
        "extensions",
        [
            pytest.param(
                [".jpg", ".png", ".gif"],
                id="mixed_extensions"
            ),
            pytest.param(
                [".JPG", ".JPEG", ".PNG"],
                id="uppercase_extensions"
            ),
            pytest.param(
                [".webp", ".heic"],
                id="modern_formats"
            ),
        ],
    )
    def test_process_photos_various_extensions(
        self, mocker, mock_anthropic_client, tmp_path, extensions
    ):
        """Test processing photos with various file extensions."""
        process_dir = tmp_path / "process"
        renamed_dir = tmp_path / "renamed"
        original_dir = tmp_path / "original"
        process_dir.mkdir()
        renamed_dir.mkdir()
        original_dir.mkdir()

        # Create test image files with different extensions
        for i, ext in enumerate(extensions):
            test_image = process_dir / f"IMG_{i:03d}{ext}"
            test_image.write_text(f"fake image data {i}")

        mock_identify = mocker.patch('rename_photos_ai.rename_photos.identify_movie')
        mock_identify.side_effect = [f"Movie_{i}" for i in range(len(extensions))]

        mock_anthropic = mocker.patch('rename_photos_ai.rename_photos.Anthropic')
        mock_anthropic.return_value = mock_anthropic_client

        process_photos(process_dir, renamed_dir, original_dir, "fake_api_key")

        # Check that all files were processed
        assert mock_identify.call_count == len(extensions)

        # Check that renamed files all have .jpg extension
        for i in range(len(extensions)):
            renamed_file = renamed_dir / f"Movie_{i}.jpg"
            assert renamed_file.exists()

    def test_process_photos_error_handling(self, mocker, mock_anthropic_client, tmp_path, capsys):
        """Test that errors during processing are handled gracefully."""
        process_dir = tmp_path / "process"
        renamed_dir = tmp_path / "renamed"
        original_dir = tmp_path / "original"
        process_dir.mkdir()
        renamed_dir.mkdir()
        original_dir.mkdir()

        test_image = process_dir / "IMG_001.jpg"
        test_image.write_text("fake image data")

        mock_identify = mocker.patch('rename_photos_ai.rename_photos.identify_movie')
        mock_identify.side_effect = Exception("API Error")

        mock_anthropic = mocker.patch('rename_photos_ai.rename_photos.Anthropic')
        mock_anthropic.return_value = mock_anthropic_client

        process_photos(process_dir, renamed_dir, original_dir, "fake_api_key")

        captured = capsys.readouterr()
        assert "Error processing" in captured.out
        assert "API Error" in captured.out
