# D6-P2-E2-A1 Dual-Backbone Feature Cache

**Status:** `E2A1_PASS`  
**Execution date:** 2026-11-16  
**Feature interface:** `d6.e2.shared_spatial_tokens.v1`  
**Rows per backbone:** 102/102, in the registered P1-A order.

## Frozen backbone identities

| Backbone | Registered model | Weight/source identity | Runtime |
|---|---|---|---|
| DINOv2 | `dinov2_vitl14_reg` | Checkpoint SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; local source-tree SHA-256 `7dd1641727d0b7b0c58f803906d03da05b8660ba8b72da5b3c5705cb9a1b4b8c` | Python 3.11.14, PyTorch 2.0.1+cu117, torchvision 0.15.2+cu117, NumPy 1.24.1 |
| DINOv3 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Snapshot revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`; weights SHA-256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179` | Python 3.11.18, PyTorch 2.14.1+cu126, Transformers 5.23.0, NumPy 2.4.6 |

The DINOv3 config and processor-config SHA-256 values are respectively `135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5` and `960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7`. Before/after identity snapshots matched for both backbones.

## Stored representations

Each image has separately L2-normalized FP32 tokens and a Boolean spatial mask:

| Array | DINOv2 | DINOv3 |
|---|---:|---:|
| CLS token | `(1024,)` | `(1024,)` |
| Register tokens | `(4, 1024)` | `(4, 1024)` |
| Spatial patch tokens | `(1024, 1024)` | `(784, 1024)` |
| Valid-patch mask | `(1024,)` | `(784,)` |

DINOv2 used the official `forward_features()` outputs `x_norm_clstoken`, `x_norm_regtokens`, and `x_norm_patchtokens`. DINOv3 used the first CLS token, next four register tokens, and remaining row-major spatial patches from `last_hidden_state`, with the registered dynamic 2D RoPE path. No additional pooling was applied. Register tokens were not mixed into the patch bag, and no image label was propagated to patches.

## Cache fingerprints and joins

| Artifact | Rows | SHA-256 |
|---|---:|---|
| DINOv2 feature manifest | 102 | `3588bab021a9cd8043d663c1d6ce46f366f89f89802fb1ac928dfae3360efddf` |
| DINOv3 feature manifest | 102 | `20e100671344422779d848f7664cdb6290d7fe4cd0e1ed1ea09f108fe0a01887` |
| Canonical input manifest | 102 | `0fca669b4f69ed2071b4f4c0ce8f60728c6bd0e6efe126be580f23037b748133` |

The two feature manifests contain the same ordered authorized image set. Both join exactly to the source SHA and canonical tensor SHA for each registered row. Per-image file and array hashes are held in the Git-ignored cache, not published here.

These representations are not asserted to be semantically equivalent across backbones. Global pooling, spatial aggregation, MIL scoring, training, and performance evaluation are outside E2-A1.
