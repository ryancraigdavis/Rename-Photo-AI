#!/usr/bin/env python3
"""
Analyze Blu-ray disc photos and rename them based on identified movie titles.
Uses Claude's vision API to identify movies from disc/case images.
"""

import base64
import os
import re
import shutil
from io import BytesIO
from pathlib import Path

from anthropic import Anthropic
from PIL import Image


def sanitize_filename(title: str) -> str:
    """
    Convert a movie title into a safe filename with Title Case and underscores.

    Args:
        title: The movie title to sanitize

    Returns:
        A filesystem-safe filename in Title_Case format
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Convert to Title Case (capitalize each word separated by underscores)
    sanitized = '_'.join(word.capitalize() for word in sanitized.split('_'))
    return sanitized


def preprocess_image(image_path: Path) -> BytesIO:
    """
    Preprocess an image for Claude API:
    - Resize to max 2048px dimension
    - Convert to RGB
    - Save as JPEG with quality=80 in memory

    Args:
        image_path: Path to the image file

    Returns:
        BytesIO object containing the preprocessed JPEG image
    """
    # Open the image
    img = Image.open(image_path)

    # Convert to RGB if needed (handles PNG with transparency, RGBA, etc.)
    if img.mode != 'RGB':
        # If image has transparency, paste it on white background
        if img.mode in ('RGBA', 'LA', 'PA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'PA':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        else:
            img = img.convert('RGB')

    # Resize if larger than 2048px in any dimension
    max_dimension = 2048
    if max(img.size) > max_dimension:
        # Calculate new size maintaining aspect ratio
        ratio = max_dimension / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Save to BytesIO as JPEG with quality=80
    output = BytesIO()
    img.save(output, format='JPEG', quality=80, optimize=True)
    output.seek(0)

    return output


def encode_image_from_bytes(image_bytes: BytesIO) -> str:
    """
    Encode an image from BytesIO to base64.

    Args:
        image_bytes: BytesIO object containing the image

    Returns:
        Base64 encoded string of the image
    """
    return base64.standard_b64encode(image_bytes.getvalue()).decode('utf-8')


def identify_movie(client: Anthropic, image_path: Path) -> str:
    """
    Use Claude's vision API to identify the movie from a Blu-ray disc/case image.

    Args:
        client: Anthropic client instance
        image_path: Path to the image file

    Returns:
        The identified movie title
    """
    print(f"Analyzing {image_path.name}...")

    # Preprocess the image
    print(f"  Preprocessing image...")
    preprocessed_image = preprocess_image(image_path)

    # Encode the preprocessed image
    image_data = encode_image_from_bytes(preprocessed_image)

    # Call Claude API with preprocessed JPEG
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "This is a photo of a Blu-ray disc or its case. Please identify the movie title. Respond with ONLY the movie title, nothing else. If you cannot identify the movie, respond with 'Unknown'."
                    }
                ],
            }
        ],
    )

    # Extract the movie title from the response
    title = message.content[0].text.strip()
    print(f"  Identified as: {title}")
    return title


def process_photos(process_dir: Path, renamed_dir: Path, original_dir: Path, api_key: str) -> None:
    """
    Process all photos in the process directory.

    Args:
        process_dir: Directory containing photos to process
        renamed_dir: Directory to save renamed photos
        original_dir: Directory to save original photos
        api_key: Anthropic API key
    """
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)

    # Get all image files from the process directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.heif'}
    image_files = [
        f for f in process_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {process_dir}")
        return

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    for image_path in sorted(image_files):
        try:
            # Identify the movie
            movie_title = identify_movie(client, image_path)

            # Sanitize the title for filename
            safe_title = sanitize_filename(movie_title)

            # Create new filename (always .jpg for renamed, keep original extension for backup)
            new_filename = f"{safe_title}.jpg"
            new_path = renamed_dir / new_filename

            # Handle duplicate filenames in renamed directory
            counter = 1
            while new_path.exists():
                new_filename = f"{safe_title}_{counter}.jpg"
                new_path = renamed_dir / new_filename
                counter += 1

            # Copy original to original_images directory with same naming scheme
            original_filename = f"{safe_title}{image_path.suffix}"
            original_path = original_dir / original_filename

            # Handle duplicate filenames in original directory
            counter = 1
            while original_path.exists():
                original_filename = f"{safe_title}_{counter}{image_path.suffix}"
                original_path = original_dir / original_filename
                counter += 1

            # Copy original file to original_images
            shutil.copy2(image_path, original_path)
            print(f"  Saved original as: {original_filename}")

            # Move processed file to renamed directory
            image_path.rename(new_path)
            print(f"  Renamed to: {new_filename}\n")

        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}\n")
            continue


def main():
    """Main entry point."""
    # Get API key from environment (Doppler will inject this)
    api_key = os.environ.get('CLAUDE_API')
    if not api_key:
        print("Error: CLAUDE_API environment variable not set")
        print("Make sure to run with Doppler: doppler run -- python src/rename_photos_ai/rename_photos.py")
        return 1

    # Set up paths
    base_dir = Path(__file__).parent
    process_dir = base_dir / 'data' / 'process'
    renamed_dir = base_dir / 'data' / 'renamed'
    original_dir = base_dir / 'data' / 'original_images'

    # Ensure directories exist
    process_dir.mkdir(parents=True, exist_ok=True)
    renamed_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Blu-ray Photo Analyzer and Renamer")
    print("=" * 60)
    print(f"Process directory: {process_dir}")
    print(f"Renamed directory: {renamed_dir}")
    print(f"Original images directory: {original_dir}")
    print("=" * 60)
    print()

    # Process all photos
    process_photos(process_dir, renamed_dir, original_dir, api_key)

    print("=" * 60)
    print("Processing complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
