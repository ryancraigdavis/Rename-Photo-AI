# Rename-Photo-AI

Automatically identify and rename Blu-ray disc/case photos using Claude's vision API.

## Overview

This Python script analyzes photos of Blu-ray discs or their cases, identifies the movie using Claude Sonnet 4.5's vision capabilities, and automatically renames the files based on the identified movie titles.

## Features

- **AI-Powered Movie Identification**: Uses Claude Sonnet 4.5 vision API to identify movies from photos
- **Smart Image Preprocessing**:
  - Automatically resizes images to max 2048px dimension
  - Converts all formats (PNG, HEIC, etc.) to RGB
  - Compresses to JPEG (quality=80) for efficient API calls
  - All preprocessing done in-memory with BytesIO
- **Original File Preservation**: Keeps untouched copies of original files
- **Clean Naming Convention**: Converts movie titles to Title_Case format (e.g., `John_Wick.jpg`)
- **Duplicate Handling**: Automatically handles duplicate filenames

## How It Works

1. **Read**: Script scans `src/rename_photos_ai/data/process/` for image files
2. **Preprocess**: Each image is resized, converted to RGB, and compressed to JPEG in memory
3. **Analyze**: Preprocessed image is sent to Claude's vision API for movie identification
4. **Backup**: Original file is copied to `src/rename_photos_ai/data/original_images/` with the identified movie name
5. **Rename**: File is moved from `process/` to `data/renamed/` with the movie title in Title_Case format

## Directory Structure

```
src/rename_photos_ai/
├── rename_photos.py          # Main script
└── data/
    ├── process/              # Put photos here to process
    ├── renamed/              # Processed photos end up here (as .jpg)
    └── original_images/      # Original files backed up here
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Doppler](https://www.doppler.com/) for secrets management
- Anthropic API key

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Doppler**:
   - Create a Doppler project named `photo-ai-rename`
   - Create a `dev` config
   - Add your Anthropic API key as `CLAUDE_API`

3. **Prepare your photos**:
   - Place photos of Blu-ray discs/cases in `src/rename_photos_ai/data/process/`

## Usage

Run the script with Doppler to inject your API key:

```bash
doppler run -- python src/rename_photos_ai/rename_photos.py
```

Or with uv:

```bash
doppler run -- uv run src/rename_photos_ai/rename_photos.py
```

## Example

**Before** (in `data/process/`):
```
IMG_0001.JPG
IMG_0002.JPG
IMG_0003.PNG
```

**After processing**:

`data/renamed/`:
```
The_Dark_Knight.jpg
Inception.jpg
Interstellar.jpg
```

`data/original_images/`:
```
The_Dark_Knight.JPG
Inception.JPG
Interstellar.PNG
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)
- HEIC/HEIF (.heic, .heif)

All output files in `renamed/` are saved as JPEG.

## Dependencies

- `anthropic>=0.40.0` - Claude API client
- `pillow>=10.0.0` - Image processing

## Notes

- The script uses Claude Sonnet 4.5 model (`claude-sonnet-4-20250514`)
- Images are preprocessed to optimize API costs and performance
- Original files are never modified or deleted
- Files are moved from `process/` to `renamed/`, so `process/` will be empty after completion