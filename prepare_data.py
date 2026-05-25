"""
Prepares training data:
- Uses existing eye_features.npy (open/awake eyes) as label=0
- Generates synthetic closed-eye images as label=1
- Saves combined eye_features.npy and labels.npy
"""
import numpy as np
import cv2

WORKSPACE = '/Users/aaryas127/Documents/GitHub/driver_drowsiness'

open_eyes = np.load(f'{WORKSPACE}/eye_features.npy')  # shape: (N, 50, 100)

def make_closed_eye(eye_img):
    """Simulate a closed eye by blending the eye with a uniform eyelid texture."""
    h, w = eye_img.shape
    eyelid_color = int(np.mean(eye_img[[0, 1, -2, -1], :]))  # skin tone from edges
    closed = np.full((h, w), eyelid_color, dtype=np.uint8)
    # Add a thin dark line (eyelash hint) in the center
    mid = h // 2
    closed[mid - 2:mid + 2, :] = max(0, eyelid_color - 30)
    # Add slight Gaussian noise for realism
    noise = np.random.normal(0, 5, closed.shape).astype(np.int16)
    closed = np.clip(closed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return closed

np.random.seed(42)
closed_eyes = np.array([make_closed_eye(eye) for eye in open_eyes])

# Slightly augment open eyes by varying brightness
augmented_open = []
for eye in open_eyes:
    factor = np.random.uniform(0.85, 1.15)
    aug = np.clip(eye.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    augmented_open.append(aug)
augmented_open = np.array(augmented_open)

all_eyes = np.concatenate([open_eyes, augmented_open, closed_eyes], axis=0)
# labels: 0=awake (open eyes), 1=drowsy (closed eyes)
all_labels = np.concatenate([
    np.zeros(len(open_eyes), dtype=np.float32),
    np.zeros(len(augmented_open), dtype=np.float32),
    np.ones(len(closed_eyes), dtype=np.float32)
])

# Shuffle
idx = np.random.permutation(len(all_eyes))
all_eyes = all_eyes[idx]
all_labels = all_labels[idx]

np.save(f'{WORKSPACE}/eye_features.npy', all_eyes)
np.save(f'{WORKSPACE}/labels.npy', all_labels)

print(f"Saved {len(all_eyes)} samples ({int(all_labels.sum())} drowsy, {int((all_labels==0).sum())} awake)")
print(f"eye_features shape: {all_eyes.shape}")
print(f"labels shape: {all_labels.shape}")
