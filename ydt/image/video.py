"""
Video processing operations for YOLO dataset creation.

Provides utilities to extract frames from video files for dataset creation.
"""

from pathlib import Path
from typing import Union, List
import concurrent.futures
import threading

import cv2

from ydt.core.logger import get_logger

logger = get_logger(__name__)


def extract_frames(
    video_path: Union[str, Path],
    frames_output_dir: Union[str, Path],
    step: int = 40,
    supported_formats: List[str] = None,
) -> int:
    """
    Extract frames from video files with specified interval.

    Can process single video file or directory containing multiple videos.
    Creates separate output directory for each video.

    Args:
        video_path: Path to video file or directory containing videos
        frames_output_dir: Root output directory for extracted frames
        step: Frame extraction interval (extract every Nth frame)
        supported_formats: List of supported video extensions

    Returns:
        Total number of frames extracted across all videos

    Raises:
        FileNotFoundError: If video path doesn't exist
        ValueError: If no supported video files found

    Examples:
        >>> # Extract from single video
        >>> extract_frames("video.mp4", "./frames", step=30)
        120

        >>> # Extract from directory of videos
        >>> extract_frames("./videos", "./output", step=40)
        450
    """
    if supported_formats is None:
        supported_formats = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".mpg", ".mpeg"]

    video_path = Path(video_path)
    frames_output_dir = Path(frames_output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    # Ensure output directory exists
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []

    if video_path.is_file():
        # Single file
        if video_path.suffix.lower() in supported_formats:
            video_files.append(video_path)
        else:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
    else:
        # Directory - find all video files
        # Use a set to avoid duplicates (rglob is case-insensitive on Windows)
        video_files_set = set()
        for ext in supported_formats:
            video_files_set.update(video_path.rglob(f"*{ext}"))
            video_files_set.update(video_path.rglob(f"*{ext.upper()}"))
        video_files = sorted(list(video_files_set))

    if not video_files:
        raise ValueError(f"No supported video files found in: {video_path}")

    logger.info(f"Found {len(video_files)} video files")

    total_saved_count = 0
    processed_videos = 0

    try:
        # Process each video file
        for video_file in video_files:
            logger.info(f"Processing video: {video_file.name}")

            # Create output directory for this video
            video_name = video_file.stem
            video_output_dir = frames_output_dir / f"{video_name}_frames"
            video_output_dir.mkdir(exist_ok=True)

            # Open video file
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                logger.warning(f"Cannot open video file: {video_file}")
                continue

            # Get video information and print
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"  Total frames: {total_frames}")
            logger.info(f"  FPS: {fps:.2f}")
            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Output directory: {video_output_dir}")

            frame_count = 0
            saved_count = 0

            try:
                # Extract frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % step == 0:
                        # Generate filename
                        output_path = video_output_dir / f"frame_{frame_count:06d}.jpg"

                        # Handle existing files
                        counter = 1
                        while output_path.exists():
                            output_path = video_output_dir / f"frame_{frame_count:06d}_{counter}.jpg"
                            counter += 1

                        # Save frame
                        cv2.imwrite(str(output_path), frame)
                        saved_count += 1

                        # Progress update
                        if saved_count % 10 == 0:
                            progress = (frame_count / total_frames) * 100
                            logger.info(
                                f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%), "
                                f"saved: {saved_count} images"
                            )

                    frame_count += 1

            finally:
                # Always release video capture
                cap.release()

            logger.info(f"  Completed: {video_file.name}")
            logger.info(f"  Processed frames: {frame_count}")
            logger.info(f"  Saved images: {saved_count}")
            total_saved_count += saved_count
            processed_videos += 1

        logger.info(f"All videos processed!")
        logger.info(f"Total images saved: {total_saved_count}")
        logger.info(f"Output directory: {frames_output_dir}")

    except KeyboardInterrupt:
        logger.warning("\nKeyboardInterrupt received! Stopping gracefully...")
        logger.info(f"Processed {processed_videos}/{len(video_files)} videos")
        logger.info(f"Total images saved before interruption: {total_saved_count}")
        raise  # Re-raise to exit properly

    return total_saved_count


def _process_single_video(video_file: Path, frames_output_dir: Path, step: int) -> int:
    """
    Process a single video file (internal function for threading).

    Args:
        video_file: Path to video file
        frames_output_dir: Root output directory
        step: Frame extraction interval

    Returns:
        Number of frames extracted from this video
    """
    # Create thread-local logger for thread safety
    thread_logger = get_logger(f"{__name__}_thread_{threading.get_ident()}")

    # Create output directory for this video
    video_name = video_file.stem
    video_output_dir = frames_output_dir / f"{video_name}_frames"
    video_output_dir.mkdir(exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        thread_logger.warning(f"Cannot open video file: {video_file}")
        return 0

    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    thread_logger.info(f"Processing: {video_file.name}")
    thread_logger.info(f"  Total frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")

    frame_count = 0
    saved_count = 0

    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            # Generate filename
            output_path = video_output_dir / f"frame_{frame_count:06d}.jpg"

            # Handle existing files
            counter = 1
            while output_path.exists():
                output_path = video_output_dir / f"frame_{frame_count:06d}_{counter}.jpg"
                counter += 1

            # Save frame
            cv2.imwrite(str(output_path), frame)
            saved_count += 1

            # Progress update (less frequent for multi-threading)
            if saved_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                thread_logger.info(f"  {video_file.name}: {frame_count}/{total_frames} ({progress:.1f}%), saved: {saved_count}")

        frame_count += 1

    # Release resources
    cap.release()

    thread_logger.info(f"Completed: {video_file.name} - Saved {saved_count} frames")
    return saved_count


def extract_frames_parallel(
    video_path: Union[str, Path],
    frames_output_dir: Union[str, Path],
    step: int = 40,
    max_workers: int = None,
    supported_formats: List[str] = None,
) -> int:
    """
    Extract frames from video files using parallel processing for multiple videos.

    Process multiple video files concurrently using thread pools for faster processing.
    Creates separate output directory for each video.

    Args:
        video_path: Path to video file or directory containing videos
        frames_output_dir: Root output directory for extracted frames
        step: Frame extraction interval (extract every Nth frame)
        max_workers: Maximum number of concurrent workers (default: CPU count)
        supported_formats: List of supported video extensions

    Returns:
        Total number of frames extracted across all videos

    Raises:
        FileNotFoundError: If video path doesn't exist
        ValueError: If no supported video files found

    Examples:
        >>> # Extract from directory with parallel processing
        >>> extract_frames_parallel("./videos", "./output", step=40, max_workers=4)
        450

        >>> # Auto-detect worker count
        >>> extract_frames_parallel("./videos", "./output", step=30)
        1200
    """
    if supported_formats is None:
        supported_formats = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".mpg", ".mpeg"]

    video_path = Path(video_path)
    frames_output_dir = Path(frames_output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    # Ensure output directory exists
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_files = []

    if video_path.is_file():
        # Single file
        if video_path.suffix.lower() in supported_formats:
            video_files.append(video_path)
        else:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
    else:
        # Directory - find all video files
        # Use a set to avoid duplicates (rglob is case-insensitive on Windows)
        video_files_set = set()
        for ext in supported_formats:
            video_files_set.update(video_path.rglob(f"*{ext}"))
            video_files_set.update(video_path.rglob(f"*{ext.upper()}"))
        video_files = sorted(list(video_files_set))

    if not video_files:
        raise ValueError(f"No supported video files found in: {video_path}")

    # If only one video file, use regular processing
    if len(video_files) == 1:
        logger.info("Single video file detected, using regular processing")
        return extract_frames(video_path, frames_output_dir, step, supported_formats)

    # Determine worker count
    import os
    if max_workers is None:
        max_workers = min(len(video_files), os.cpu_count() or 4)

    max_workers = min(max_workers, len(video_files))  # Don't exceed video count
    max_workers = max(1, max_workers)  # At least 1 worker

    logger.info(f"Found {len(video_files)} video files")
    logger.info(f"Using {max_workers} parallel workers for processing")

    total_saved_count = 0

    # Process videos in parallel
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Submit all video processing tasks
        future_to_video = {
            executor.submit(_process_single_video, video_file, frames_output_dir, step): video_file
            for video_file in video_files
        }

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_video):
            video_file = future_to_video[future]
            completed_count += 1

            try:
                saved_count = future.result()
                total_saved_count += saved_count
                logger.info(f"Progress: {completed_count}/{len(video_files)} videos completed")
            except Exception as e:
                logger.error(f"Error processing {video_file.name}: {e}")

        logger.info(f"All videos processed in parallel!")
        logger.info(f"Total images saved: {total_saved_count}")
        logger.info(f"Output directory: {frames_output_dir}")

    except KeyboardInterrupt:
        logger.warning("\nKeyboardInterrupt received! Cancelling remaining tasks...")

        # Cancel all pending futures
        for future in future_to_video.keys():
            future.cancel()

        # Shutdown executor immediately without waiting
        executor.shutdown(wait=False, cancel_futures=True)

        logger.info(f"Cancelled. Processed {completed_count}/{len(video_files)} videos")
        logger.info(f"Total images saved before cancellation: {total_saved_count}")

        raise  # Re-raise to exit properly

    finally:
        # Ensure executor is properly shut down
        executor.shutdown(wait=True)

    return total_saved_count
