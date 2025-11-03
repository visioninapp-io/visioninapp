# PyTorch 2.6 Compatibility Fix

## Problem

PyTorch 2.6 changed the default `weights_only=True` in `torch.load()` for security. This breaks Ultralytics YOLO models:

```
WeightsUnpickler error: Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel 
was not an allowed global by default.
```

## Solutions

### Option 1: Upgrade Ultralytics (Recommended) ✅

The latest Ultralytics (8.3+) supports PyTorch 2.6!

```bash
cd AI
pip install --upgrade ultralytics
```

**Or update `AI/requirements.txt`:**
```
ultralytics>=8.3.0
```

### Option 2: Add Safe Globals (Already Implemented) ✅

Already added to `AI/training_service.py`:

```python
# Fix PyTorch 2.6 weights_only issue
import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception as e:
        logger.warning(f"Could not add safe globals: {e}")
```

### Option 3: Wait for Ultralytics Update

Ultralytics will likely update to be compatible with PyTorch 2.6. Check:
```bash
pip install ultralytics --upgrade
```

## Recommended Action

**Upgrade Ultralytics to latest version** (supports PyTorch 2.6):

```bash
# AI Server
cd AI
./INSTALL_FIX.sh
# or manually:
pip install --upgrade ultralytics
```

## Status Sync Fix

Also fixed status synchronization from AI service to Backend:

**Changes:**
1. AI service now properly sets `"status": "failed"` with error message
2. Backend syncs failed status from AI service
3. Frontend polling will now show failed status

**Before:**
- AI service fails
- Backend still shows "running"
- Frontend keeps polling with no update

**After:**
- AI service fails → sets status to "failed"
- Backend polls AI service → syncs "failed" status
- Frontend shows error message to user

## Test the Fix

```bash
# 1. Upgrade Ultralytics
cd AI
./INSTALL_FIX.sh

# 2. Restart AI service
./start.sh

# 3. Try training again
# Should work now with PyTorch 2.6 + Ultralytics 8.3+!
```

