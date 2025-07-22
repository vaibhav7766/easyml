# Virtual Environment Setup

This project has two virtual environments:

## Current Active Environment (Recommended)
- **Location**: `.direnv/python-3.10.12/`
- **Managed by**: VS Code Python environment system
- **Size**: ~595MB
- **Status**: ✅ Active and configured
- **Dependencies**: Up-to-date with requirements.txt

## Legacy Environment (Backup)
- **Location**: `.venv/`
- **Managed by**: Traditional venv
- **Size**: ~439MB
- **Status**: ⚠️ Inactive but preserved
- **Purpose**: Backup/compatibility

## Usage Instructions

### For Development (Current)
```bash
# Already activated automatically
python -m app.main
```

### To Switch to Legacy Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

### To Remove Legacy Environment (Optional)
```bash
# Only if you're certain you don't need it
rm -rf .venv/
```

## Note
The `.direnv/` environment is recommended as it's actively maintained and has the latest dependencies. The `.venv/` is kept for compatibility and as a backup.
