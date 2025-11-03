"""
DEPRECATED - This file is no longer used

All training now happens via AI Service (AI/main.py)

Previous functionality:
- PyTorch classification model training (ResNet, MobileNet)
- Training on CPU in BE server

Current approach:
- ALL training (YOLO + PyTorch) should use AI Service
- Training on GPU in dedicated AI server

To remove this file:
1. Ensure AI service supports PyTorch models (or remove PyTorch architectures from frontend)
2. Update any remaining references
3. Delete this file

Migration date: 2024-10-30
"""

# This file intentionally empty - kept for reference only
# See: BE/app/services/ai_client.py for new implementation

