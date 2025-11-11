"""
Test CLI interface functionality
"""

import subprocess
import sys
from pathlib import Path
import pytest
import yaml


class TestCLIInterface:
    """Test CLI commands"""

    def run_cli_command(self, args, cwd=None):
        """Helper to run CLI command"""
        cmd = [sys.executable, "-m", "ydt.cli.main"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)
        return result

    def test_cli_help(self):
        """Test CLI help command"""
        result = self.run_cli_command(["--help"])
        assert result.returncode == 0
        assert "YOLO Dataset Tools" in result.stdout
        assert "image" in result.stdout
        assert "dataset" in result.stdout

    def test_cli_version(self):
        """Test CLI version command"""
        result = self.run_cli_command(["--version"])
        assert result.returncode == 0
        assert "ydt 0.2.5" in result.stdout

    def test_image_help(self):
        """Test image module help"""
        result = self.run_cli_command(["image", "--help"])
        assert result.returncode == 0
        assert "slice" in result.stdout
        assert "resize" in result.stdout
        assert "video" in result.stdout

    def test_video_help(self):
        """Test video extraction help"""
        result = self.run_cli_command(["image", "video", "--help"])
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "--step" in result.stdout

    def test_video_extraction_single_file(self, sample_video, temp_dir):
        """Test video extraction from single file"""
        output_dir = temp_dir / "frames"

        result = self.run_cli_command(
            [
                "image",
                "video",
                "-i",
                str(sample_video),
                "-o",
                str(output_dir),
                "-s",
                "5",  # Extract every 5th frame
            ]
        )

        assert result.returncode == 0
        assert output_dir.exists()

        # Check if frames were extracted
        frames = list(output_dir.rglob("*.jpg"))
        assert len(frames) > 0  # Should extract at least 1 frame

    def test_video_extraction_directory(self, sample_video, temp_dir):
        """Test video extraction from directory"""
        video_dir = temp_dir / "videos"
        output_dir = temp_dir / "frames"
        video_dir.mkdir()

        # Copy video to directory
        import shutil

        shutil.copy(sample_video, video_dir / "video1.mp4")

        result = self.run_cli_command(
            ["image", "video", "-i", str(video_dir), "-o", str(output_dir), "-s", "5"]
        )

        assert result.returncode == 0
        assert output_dir.exists()

        # Check for video-specific output directory
        video_output_dir = output_dir / "video1_frames"
        assert video_output_dir.exists()
        frames = list(video_output_dir.rglob("*.jpg"))
        assert len(frames) > 0

    def test_dataset_help(self):
        """Test dataset module help"""
        result = self.run_cli_command(["dataset", "--help"])
        assert result.returncode == 0
        assert "split" in result.stdout
        assert "merge" in result.stdout
        assert "synthesize" in result.stdout

    def test_dataset_split_help(self):
        """Test dataset split help"""
        result = self.run_cli_command(["dataset", "split", "--help"])
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "--ratio" in result.stdout

    def test_dataset_split(self, sample_dataset, temp_dir):
        """Test dataset splitting"""
        output_dir = temp_dir / "split_dataset"

        result = self.run_cli_command(
            [
                "dataset",
                "split",
                "-i",
                str(sample_dataset / "data.yaml"),
                "-o",
                str(output_dir),
                "-r",
                "0.8",
            ]
        )

        assert result.returncode == 0
        assert output_dir.exists()

        # Check if split directories were created
        train_dir = output_dir / "images" / "train"
        val_dir = output_dir / "images" / "val"
        assert train_dir.exists()
        assert val_dir.exists()

    def test_viz_help(self):
        """Test visualization module help"""
        result = self.run_cli_command(["viz", "--help"])
        assert result.returncode == 0
        assert "dataset" in result.stdout
        assert "letterbox" in result.stdout
        assert "augment" in result.stdout

    def test_nonexistent_command(self):
        """Test error handling for nonexistent command"""
        result = self.run_cli_command(["nonexistent"])
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_missing_required_args(self):
        """Test error handling for missing required arguments"""
        result = self.run_cli_command(["image", "video"])
        assert result.returncode != 0
        assert "required" in result.stderr.lower()

    def test_invalid_input_file(self, temp_dir):
        """Test error handling for invalid input file"""
        output_dir = temp_dir / "output"

        result = self.run_cli_command(
            ["image", "video", "-i", "nonexistent.mp4", "-o", str(output_dir)]
        )

        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "doesn't exist" in result.stderr.lower()


class TestCLIWithYDTCommand:
    """Test with actual ydt executable (if available)"""

    def setup_method(self):
        """Check if ydt command is available"""
        import shutil

        self.ydt_available = shutil.which("ydt") is not None

    def test_ydt_command_if_available(self):
        """Test actual ydt command if it's installed"""
        if not self.ydt_available:
            pytest.skip("ydt command not available")

        result = subprocess.run(["ydt", "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "ydt 0.2.5" in result.stdout
