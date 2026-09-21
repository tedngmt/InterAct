# GRAB → Unity export

Converts processed GRAB sequences into Unity-ready JSON clips, in two variants:

| Mode | Contents |
|---|---|
| `fullbody` | Whole SMPL-X skeleton (55 joints: body, jaw/eyes, both hands) + the interacting object, in world space with the floor at `y = 0`. |
| `hands` | Left and right hand only — wrist + 15 finger joints each, the MANO layout — expressed in that hand's own wrist frame, with the object carried into the same frame. |

## Prerequisites

The InterAct GRAB pipeline must have been run first, so that
`data/grab/sequences_canonical/` exists. SMPL-X models must be in `./models/smplx/`.

```bash
conda activate interact
```

## Usage

```bash
# Full body
python unity_export/export_grab_unity.py --mode fullbody

# Hands only (wrist-local)
python unity_export/export_grab_unity.py --mode hands
```

Output lands in `data/grab/unity/<mode>/<sequence>.json`, one file per sequence.

Useful flags: `--src` (input folder, defaults to `sequences_canonical`), `--out`,
`--limit N` (export only the first N sequences), `--indent 2` (pretty-print).

Object meshes are **not** copied — reference `data/grab/objects/<name>/<name>.obj`,
which each clip names in its `object.mesh` field.

## Coordinate conventions

GRAB data is right-handed Y-up; Unity is **left-handed Y-up**. The exporter
converts by mirroring Z:

- positions `(x, y, z)` → `(x, y, -z)`
- quaternions `(x, y, z, w)` → `(-x, -y, z, w)`

So the JSON is already in Unity's frame — do not convert again on import.

Clips are **30 fps** (`process_grab.py` downsamples GRAB's native 120 fps by
`[::4]`). The floor is at `y = 0`: `process_grab.py` subtracts the body's minimum
height over the first 30 frames.

## JSON schema

Common fields:

```jsonc
{
  "name": "s1_apple_lift_0",
  "fps": 30,
  "frameCount": 432,
  "subject": "s1",
  "gender": "male",
  "mode": "fullbody",          // or "hands"
  "text": "A person lifts the apple ..."
}
```

`fullbody` adds:

```jsonc
{
  "skeleton": {
    "jointNames":  ["pelvis", "left_hip", ...],   // 55 entries
    "parents":     [-1, 0, 0, ...],               // -1 for the root
    "restLocalPositions": [[x, y, z], ...]        // parent-relative rest offsets
  },
  "rootPositions":  [[x, y, z], ...],             // per frame
  "localRotations": [[[x, y, z, w], ...], ...],   // [frame][joint], parent-relative
  "object": {
    "name": "apple",
    "mesh": "apple.obj",
    "positions": [[x, y, z], ...],
    "rotations": [[x, y, z, w], ...]
  }
}
```

`hands` replaces those with `hands.left` and `hands.right`, each holding its own
`skeleton`, `localRotations` and `object`. The wrist is joint 0, sits at the
origin, and carries the identity rotation — the hand is already in its own frame,
so the arm's motion through space is deliberately discarded. Finger rotations are
parent-relative and identical to their full-body counterparts.

Rotations are **local (parent-relative)**, so plain forward kinematics
reconstructs the pose:

```
globalRotation[j] = globalRotation[parent[j]] * localRotation[j]
globalPosition[j] = globalPosition[parent[j]] + globalRotation[parent[j]] * restLocal[j]
```

## Unity setup

1. Install **Newtonsoft Json** (Package Manager → Add package by name →
   `com.unity.nuget.newtonsoft-json`). The loader uses it for the nested arrays,
   which Unity's built-in `JsonUtility` cannot express.
2. Copy `unity/GrabClip.cs` and `unity/GrabPlayer.cs` into your project's `Assets/`.
3. Add `GrabPlayer` to an empty GameObject, set `clipPath` to an exported `.json`,
   and optionally assign `objectPrefab` (otherwise a small cube stands in for the
   object). Press play.

`GrabPlayer` builds the joint hierarchy from `skeleton`, applies
`restLocalPositions` once as local offsets, then drives `localRotation` per frame —
the same maths as the block above. It handles both modes, spawning one hierarchy
per hand for `hands` clips.

## Body shape caveat

GRAB encodes body shape as a per-subject **`v_template`** — a full 10 475-vertex
mesh stored in `human.npz` — not as the 10–16 shape *betas* that MPI's SMPL-X
Unity plugin expects. The exported skeleton already reflects each subject's
template, because rest joints are computed as `J_regressor @ v_template`, so
**joint positions and motion are exact**.

What this affects is skinning: to render the correct body *surface* you need the
subject's template mesh rather than a beta-shaped default body. There are ten
subjects, so the practical options are to export the ten templates as meshes, or
to fit betas per subject once. For `hands` mode this matters much less, since the
wrist-local finger motion is what carries the signal.

## Verification

The exporter's forward kinematics was checked against `smplx`'s own forward pass:
joint positions agree to **3.5 × 10⁻⁷ m**, i.e. the float32 noise floor of the
reference. Reconstructing the rig from the written JSON alone — the same path
`GrabPlayer` takes — agrees to **4.4 × 10⁻⁶ m**, matching the 6-decimal rounding
in the files. The wrist-local object track preserves object-to-wrist distance to
**3 × 10⁻⁶ m**, confirming it is a rigid change of frame.
