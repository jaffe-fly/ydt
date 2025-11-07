# 📦 Publishing Guide

Complete guide for building and publishing YDT to PyPI.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Versioning](#versioning)
- [Building the Package](#building-the-package)
- [Testing the Build](#testing-the-build)
- [Publishing to PyPI](#publishing-to-pypi)
- [Publishing Checklist](#publishing-checklist)
- [Continuous Integration](#continuous-integration)

---

## Prerequisites

### Tools Required

1. **Build tools**
```bash
pip install build twine
```

2. **PyPI Account**
- Create account at [pypi.org](https://pypi.org/account/register/)
- Create account at [test.pypi.org](https://test.pypi.org/account/register/) for testing

3. **API Tokens**
- Generate PyPI API token: https://pypi.org/manage/account/token/
- Generate TestPyPI token: https://test.pypi.org/manage/account/token/

### Configure Credentials

Create `~/.pypirc` (Linux/macOS) or `%USERPROFILE%\.pypirc` (Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

**Security Note**: Keep this file secure (chmod 600 on Linux/macOS)

---

## Versioning

YDT follows [Semantic Versioning](https://semver.org/) (SemVer):

```
MAJOR.MINOR.PATCH

Example: 1.2.3
- MAJOR (1): Breaking changes
- MINOR (2): New features (backward compatible)
- PATCH (3): Bug fixes
```

### Update Version

Edit `pyproject.toml`:

```toml
[project]
name = "ydt"
version = "0.2.0"  # Update this
```

And `__init__.py`:

```python
__version__ = "0.2.0"  # Update this
```

### Version Examples

| Change Type | Old | New | Example |
|-------------|-----|-----|---------|
| Bug fix | 1.0.0 | 1.0.1 | Fixed label parsing |
| New feature | 1.0.1 | 1.1.0 | Added new command |
| Breaking change | 1.1.0 | 2.0.0 | Changed API |

---

## Building the Package

### Step 1: Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info
# Windows
rmdir /s /q dist build
del /s /q *.egg-info
```

### Step 2: Update Version

1. Update version in `pyproject.toml`
2. Update version in `__init__.py`
3. Update version in `cli/main.py` if hardcoded

### Step 3: Update Changelog

Add entry to `CHANGELOG.md`:

```markdown
## [0.2.0] - 2025-01-15

### Added
- New synthetic dataset generation feature
- Support for custom augmentation pipelines

### Fixed
- Fixed label transformation bug in OBB rotation
- Improved memory usage in large dataset processing

### Changed
- Updated CLI interface for better usability
```

### Step 4: Build

```bash
# Using build (recommended)
python -m build

# This creates:
# - dist/ydt-0.2.0.tar.gz (source distribution)
# - dist/ydt-0.2.0-py3-none-any.whl (wheel)
```

### Step 5: Verify Build

```bash
# Check contents of wheel
unzip -l dist/ydt-0.2.0-py3-none-any.whl

# Should show:
# - cli/
# - core/
# - dataset/
# - image/
# - quality/
# - visual/
# - __init__.py
# - pyproject.toml
```

---

## Testing the Build

### Test 1: Local Installation

```bash
# Create fresh venv
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install from wheel
pip install dist/ydt-0.2.0-py3-none-any.whl

# Test import
python -c "from image import slice_dataset; print('Success!')"

# Test CLI
ydt --version

# Clean up
deactivate
rm -rf test_env
```

### Test 2: TestPyPI

Upload to TestPyPI first:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ydt

# Test
ydt --version
```

**Note**: Use `--extra-index-url` because TestPyPI doesn't have all dependencies.

---

## Publishing to PyPI

### Pre-Publishing Checklist

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated
- [ ] Version number is correct
- [ ] LICENSE file exists
- [ ] README.md is polished
- [ ] Tested on TestPyPI
- [ ] Git tag created

### Create Git Tag

```bash
# Commit all changes
git add .
git commit -m "Release v0.2.0"

# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0"

# Push tag
git push origin v0.2.0
```

### Upload to PyPI

```bash
# Upload to PyPI
twine upload dist/*

# You'll be asked for credentials (or use API token)
```

### Verify Publication

```bash
# Install from PyPI
pip install ydt

# Check version
ydt --version

# Visit package page
# https://pypi.org/project/ydt/
```

---

## Publishing Checklist

### Before Building

- [ ] Update version in `pyproject.toml`
- [ ] Update version in `__init__.py`
- [ ] Update version in `cli/main.py`
- [ ] Update `CHANGELOG.md`
- [ ] Update `README.md` if needed
- [ ] Run all tests: `pytest`
- [ ] Run linter: `ruff check .`
- [ ] Run type checker: `mypy .`
- [ ] Format code: `black .`

### Building

- [ ] Clean old builds: `rm -rf dist/`
- [ ] Build package: `python -m build`
- [ ] Verify wheel contents
- [ ] Test local installation

### Testing

- [ ] Upload to TestPyPI
- [ ] Install from TestPyPI
- [ ] Test all commands
- [ ] Test Python API
- [ ] Check documentation rendering

### Publishing

- [ ] Create git tag
- [ ] Push tag to GitHub
- [ ] Upload to PyPI
- [ ] Verify installation from PyPI
- [ ] Create GitHub release
- [ ] Announce release

---

## Continuous Integration

### GitHub Actions Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### Set GitHub Secrets

1. Go to repository Settings → Secrets
2. Add `PYPI_API_TOKEN` with your PyPI API token

### Create Release

```bash
# Tag and push
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0

# Create GitHub release
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes here"
```

This triggers automatic publishing to PyPI.

---

## Advanced Topics

### Building for Multiple Platforms

```bash
# Build source distribution
python -m build --sdist

# Build wheel
python -m build --wheel

# Build both
python -m build
```

### Custom Build Scripts

Create `build.sh`:

```bash
#!/bin/bash
set -e

echo "Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

echo "Running tests..."
pytest

echo "Linting..."
ruff check .

echo "Type checking..."
mypy .

echo "Building package..."
python -m build

echo "Build complete!"
ls -lh dist/
```

### Version Automation

Use `bumpversion` to automate version updates:

```bash
pip install bump2version

# Bump patch version (0.1.0 -> 0.1.1)
bumpversion patch

# Bump minor version (0.1.1 -> 0.2.0)
bumpversion minor

# Bump major version (0.2.0 -> 1.0.0)
bumpversion major
```

---

## Troubleshooting

### Issue: "File already exists"

**Problem**: Trying to upload same version twice.

**Solution**: Increment version number and rebuild.

### Issue: "Invalid distribution"

**Problem**: Build artifacts are corrupted.

**Solution**:
```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

### Issue: "Authentication failed"

**Problem**: Wrong API token or credentials.

**Solution**: Check `~/.pypirc` and regenerate API token if needed.

### Issue: "Missing dependencies on install"

**Problem**: Dependencies not specified correctly.

**Solution**: Check `pyproject.toml` dependencies list.

---

## Best Practices

### 1. Test Before Publishing

Always test on TestPyPI first:

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ ydt
```

### 2. Use Pre-release Versions

For alpha/beta releases:

```toml
[project]
version = "1.0.0a1"  # Alpha
version = "1.0.0b1"  # Beta
version = "1.0.0rc1" # Release candidate
```

### 3. Keep Good Records

Maintain detailed `CHANGELOG.md`:

```markdown
## [Unreleased]
### Added
- Feature in development

## [1.0.0] - 2025-01-15
### Added
- Initial release
```

### 4. Semantic Versioning

Follow SemVer strictly:
- PATCH for bug fixes
- MINOR for new features
- MAJOR for breaking changes

---

## Publishing Workflow Summary

```bash
# 1. Update version
vim pyproject.toml
vim __init__.py

# 2. Update changelog
vim CHANGELOG.md

# 3. Run tests
pytest

# 4. Build
rm -rf dist/
python -m build

# 5. Test locally
pip install dist/*.whl

# 6. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 7. Test from TestPyPI
pip install --index-url https://test.pypi.org/simple/ ydt

# 8. Create git tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 9. Upload to PyPI
twine upload dist/*

# 10. Verify
pip install ydt
```

---

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI](https://pypi.org/)
- [TestPyPI](https://test.pypi.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

---

[⬆ Back to Top](#-publishing-guide) | [⬅️ Back: API Reference](api-reference.md)
