"""
Test CLI interface functionality
"""

import subprocess
import sys


class TestCLIInterface:
    """Test CLI commands"""

    def run_cli_command(self, args, cwd=None):
        """Helper to run CLI command"""
        cmd = [sys.executable, "-m", "ydt.cli.main"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            timeout=30,
        )
        return result

    def test_cli_version(self):
        """Test CLI version command"""
        result = self.run_cli_command(["--version"])
        assert result.returncode == 0
        # Check for version number (flexible check)
        assert "ydt" in result.stdout.lower()

    def test_video_extraction_single_file(self, sample_video, temp_dir):
        """Test video extraction from single file"""
        output_dir = temp_dir / "frames"

        result = self.run_cli_command(
            [
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

    def test_dataset_split(self, sample_dataset, temp_dir):
        """Test dataset splitting"""
        output_dir = temp_dir / "split_dataset"

        result = self.run_cli_command(
            [
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

    def test_analyze_dataset(self, sample_dataset):
        """Test dataset analysis command"""
        result = self.run_cli_command(
            [
                "analyze",
                "-i",
                str(sample_dataset),
                "--split",
                "train",
            ]
        )

        assert result.returncode == 0
        # Check that analysis output is generated
        assert "Analysis completed" in result.stdout or "数据集分析" in result.stdout

    def test_nonexistent_command(self):
        """Test error handling for nonexistent command"""
        result = self.run_cli_command(["nonexistent"])
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_missing_required_args(self):
        """Test error handling for missing required arguments"""
        result = self.run_cli_command(["video"])
        assert result.returncode != 0
        assert "required" in result.stderr.lower()
