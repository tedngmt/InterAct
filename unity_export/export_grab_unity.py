"""Export processed GRAB sequences to Unity-friendly JSON.

Two variants are produced by ``--mode``:

``fullbody``
    The whole SMPL-X skeleton (55 joints: body, jaw/eyes, both hands) plus the
    interacting object's transform, in world space with the floor at ``y = 0``.

``hands``
    Left and right hand only (wrist + 15 finger joints each, i.e. the MANO
    layout), expressed in that hand's own wrist frame.  The object is carried
    into the same wrist-local frame so the grasp stays meaningful.

Rotations are stored as local (parent-relative) quaternions, so a Unity rig can
consume them directly with plain forward kinematics.  Everything is converted
from the dataset's right-handed Y-up frame into Unity's left-handed Y-up frame.
"""

import argparse
import json
import os

import numpy as np
import smplx
import torch

# SMPL-X joint layout (see smplx.joint_names.JOINT_NAMES).
NUM_JOINTS = 55
L_WRIST, R_WRIST = 20, 21
L_HAND = list(range(25, 40))
R_HAND = list(range(40, 55))

FPS = 30  # process_grab.py downsamples GRAB's native 120 fps by [::4]


def rotvec_to_quat(rotvec):
    """(..., 3) axis-angle -> (..., 4) quaternion in (x, y, z, w) order."""
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    # np.sinc(x) == sin(pi x) / (pi x), which stays finite as theta -> 0.
    half = 0.5 * theta
    scale = 0.5 * np.sinc(half / np.pi)
    return np.concatenate([rotvec * scale, np.cos(half)], axis=-1)


def quat_to_mat(q):
    """(..., 4) quaternion (x, y, z, w) -> (..., 3, 3) rotation matrix."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m):
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion (x, y, z, w).

    Uses the branch with the largest denominator for numerical stability.
    """
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.empty(m.shape[:-2] + (4,), dtype=m.dtype)

    big = t > 0
    s = np.sqrt(np.maximum(t + 1.0, 1e-12)) * 2
    q[big, 3] = 0.25 * s[big]
    q[big, 0] = ((m[..., 2, 1] - m[..., 1, 2]) / s)[big]
    q[big, 1] = ((m[..., 0, 2] - m[..., 2, 0]) / s)[big]
    q[big, 2] = ((m[..., 1, 0] - m[..., 0, 1]) / s)[big]

    # Fall back to the dominant-diagonal branches where the trace is <= 0.
    rest = ~big
    if np.any(rest):
        idx = np.argmax(np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1), -1)
        for k in range(3):
            sel = rest & (idx == k)
            if not np.any(sel):
                continue
            a, b = (k + 1) % 3, (k + 2) % 3
            s = np.sqrt(np.maximum(1.0 + m[..., k, k] - m[..., a, a] - m[..., b, b], 1e-12)) * 2
            q[sel, k] = 0.25 * s[sel]
            q[sel, a] = ((m[..., a, k] + m[..., k, a]) / s)[sel]
            q[sel, b] = ((m[..., b, k] + m[..., k, b]) / s)[sel]
            q[sel, 3] = ((m[..., b, a] - m[..., a, b]) / s)[sel]
    return q


def to_unity_pos(p):
    """Right-handed Y-up -> Unity's left-handed Y-up (mirror Z)."""
    out = np.array(p, dtype=np.float64, copy=True)
    out[..., 2] *= -1
    return out


def to_unity_quat(q):
    """Mirroring Z conjugates the rotation: (x, y, z, w) -> (-x, -y, z, w)."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 0] *= -1
    out[..., 1] *= -1
    return out


def forward_kinematics(local_quat, rest_local, root_pos):
    """Accumulate local rotations down the chain.

    ``local_quat`` (T, J, 4), ``rest_local`` (J, 3) parent-relative rest offsets,
    ``root_pos`` (T, 3).  Returns global rotations (T, J, 3, 3) and global
    positions (T, J, 3).
    """
    T, J = local_quat.shape[:2]
    local_mat = quat_to_mat(local_quat)
    g_rot = np.empty((T, J, 3, 3))
    g_pos = np.empty((T, J, 3))

    g_rot[:, 0] = local_mat[:, 0]
    g_pos[:, 0] = root_pos
    for j in range(1, J):
        p = PARENTS[j]
        g_rot[:, j] = g_rot[:, p] @ local_mat[:, j]
        g_pos[:, j] = g_pos[:, p] + np.einsum('tab,b->ta', g_rot[:, p], rest_local[j])
    return g_rot, g_pos


def build_full_pose(poses, model):
    """(T, 114) GRAB pose -> (T, 55, 3) local axis-angle for every SMPL-X joint.

    GRAB stores the hands as 24 PCA coefficients per side; expand them back to
    per-joint axis-angle exactly the way smplx does internally.
    """
    T = poses.shape[0]
    global_orient = poses[:, :3]
    body_pose = poses[:, 3:66]
    l_pca, r_pca = poses[:, 66:90], poses[:, 90:114]

    l_comp = model.left_hand_components.detach().cpu().numpy()
    r_comp = model.right_hand_components.detach().cpu().numpy()
    l_mean = model.left_hand_mean.detach().cpu().numpy()
    r_mean = model.right_hand_mean.detach().cpu().numpy()

    l_aa = l_pca @ l_comp + (0.0 if model.flat_hand_mean else l_mean)
    r_aa = r_pca @ r_comp + (0.0 if model.flat_hand_mean else r_mean)

    # Jaw and the two eyeballs are never animated in GRAB.
    face = np.zeros((T, 9))
    full = np.concatenate([global_orient, body_pose, face, l_aa, r_aa], axis=1)
    return full.reshape(T, NUM_JOINTS, 3)


def rest_joints(model, vtemp):
    """Rest-pose joint positions for this subject's body template."""
    reg = model.J_regressor.detach().cpu().numpy()
    return reg @ np.asarray(vtemp, dtype=np.float64)


def load_sequence(seq_dir):
    with np.load(os.path.join(seq_dir, 'human.npz'), allow_pickle=True) as f:
        human = {k: f[k] for k in f.files}
    with np.load(os.path.join(seq_dir, 'object.npz'), allow_pickle=True) as f:
        obj = {k: f[k] for k in f.files}
    text = ''
    text_path = os.path.join(seq_dir, 'text.txt')
    if os.path.exists(text_path):
        with open(text_path, encoding='utf-8') as fh:
            # Each line packs "sentence#pos-tags#start#end"; keep the sentences.
            text = ' '.join(l.split('#')[0].strip() for l in fh if l.strip())
    return human, obj, text


def export_fullbody(name, human, obj, text, g_rot, g_pos, local_quat, rest_local, subject):
    obj_quat = rotvec_to_quat(np.asarray(obj['angles'], dtype=np.float64))
    return {
        'name': name,
        'fps': FPS,
        'frameCount': int(local_quat.shape[0]),
        'subject': subject,
        'gender': str(human['gender']),
        'coordinateSystem': 'unity (left-handed, Y up, floor at y=0)',
        'mode': 'fullbody',
        'skeleton': {
            'jointNames': JOINT_NAMES[:NUM_JOINTS],
            'parents': [int(p) for p in PARENTS],
            'restLocalPositions': np.round(to_unity_pos(rest_local), 6).tolist(),
        },
        'rootPositions': np.round(to_unity_pos(g_pos[:, 0]), 6).tolist(),
        'localRotations': np.round(to_unity_quat(local_quat), 6).tolist(),
        'object': {
            'name': str(obj['name']),
            'mesh': '%s.obj' % str(obj['name']),
            'positions': np.round(to_unity_pos(np.asarray(obj['trans'], dtype=np.float64)), 6).tolist(),
            'rotations': np.round(to_unity_quat(obj_quat), 6).tolist(),
        },
        'text': text,
    }


def export_hands(name, human, obj, text, g_rot, g_pos, local_quat, rest_local, subject):
    """Wrist-local export: one entry per hand, object carried into the same frame."""
    obj_pos = np.asarray(obj['trans'], dtype=np.float64)
    obj_mat = quat_to_mat(rotvec_to_quat(np.asarray(obj['angles'], dtype=np.float64)))

    hands = {}
    for side, wrist, fingers in (('left', L_WRIST, L_HAND), ('right', R_WRIST, R_HAND)):
        joints = [wrist] + fingers
        # Re-index the chain so the wrist becomes a root at the origin.
        remap = {j: i for i, j in enumerate(joints)}
        parents = [-1] + [remap[PARENTS[j]] for j in fingers]

        w_rot, w_pos = g_rot[:, wrist], g_pos[:, wrist]
        w_rot_t = np.transpose(w_rot, (0, 2, 1))

        # The wrist itself is the identity root; fingers keep their local rotations.
        q = np.concatenate([
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (local_quat.shape[0], 1, 1)),
            local_quat[:, fingers],
        ], axis=1)

        hands[side] = {
            'skeleton': {
                'jointNames': [JOINT_NAMES[j] for j in joints],
                'parents': parents,
                'restLocalPositions': np.round(
                    to_unity_pos(np.stack([np.zeros(3)] + [rest_local[j] for j in fingers])), 6
                ).tolist(),
            },
            'localRotations': np.round(to_unity_quat(q), 6).tolist(),
            'object': {
                'name': str(obj['name']),
                'mesh': '%s.obj' % str(obj['name']),
                'positions': np.round(to_unity_pos(
                    np.einsum('tab,tb->ta', w_rot_t, obj_pos - w_pos)), 6).tolist(),
                'rotations': np.round(to_unity_quat(
                    mat_to_quat(w_rot_t @ obj_mat)), 6).tolist(),
            },
        }

    return {
        'name': name,
        'fps': FPS,
        'frameCount': int(local_quat.shape[0]),
        'subject': subject,
        'gender': str(human['gender']),
        'coordinateSystem': 'unity (left-handed, Y up), wrist-local per hand',
        'mode': 'hands',
        'hands': hands,
        'text': text,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['fullbody', 'hands'], required=True)
    ap.add_argument('--src', default='./data/grab/sequences_canonical',
                    help='folder of processed GRAB sequences')
    ap.add_argument('--out', default=None,
                    help='output folder (default: ./data/grab/unity/<mode>)')
    ap.add_argument('--models', default='./models')
    ap.add_argument('--limit', type=int, default=0, help='export at most N sequences')
    ap.add_argument('--indent', type=int, default=0,
                    help='JSON indent; 0 writes compact files')
    args = ap.parse_args()

    out_dir = args.out or os.path.join('./data/grab/unity', args.mode)
    os.makedirs(out_dir, exist_ok=True)

    models = {}

    def get_model(gender):
        if gender not in models:
            models[gender] = smplx.create(args.models, model_type='smplx', gender=gender,
                                          num_pca_comps=24, use_pca=True, batch_size=1)
        return models[gender]

    global PARENTS, JOINT_NAMES
    from smplx.joint_names import JOINT_NAMES as _NAMES
    JOINT_NAMES = list(_NAMES)
    PARENTS = get_model('male').parents.detach().cpu().numpy()

    names = sorted(os.listdir(args.src))
    if args.limit:
        names = names[:args.limit]

    written = 0
    for name in names:
        seq_dir = os.path.join(args.src, name)
        if not os.path.isdir(seq_dir):
            continue
        human, obj, text = load_sequence(seq_dir)
        gender = str(human['gender'])
        model = get_model(gender)

        poses = np.asarray(human['poses'], dtype=np.float64)
        trans = np.asarray(human['trans'], dtype=np.float64)

        full_aa = build_full_pose(poses, model)
        local_quat = rotvec_to_quat(full_aa)

        j_rest = rest_joints(model, human['vtemp'])
        rest_local = j_rest.copy()
        rest_local[1:] -= j_rest[PARENTS[1:]]
        rest_local[0] = 0.0

        # SMPL-X applies translation on top of the rest-pose root position.
        root_pos = trans + j_rest[0]
        g_rot, g_pos = forward_kinematics(local_quat, rest_local, root_pos)

        subject = name.split('_')[0]
        fn = export_fullbody if args.mode == 'fullbody' else export_hands
        payload = fn(name, human, obj, text, g_rot, g_pos, local_quat, rest_local, subject)

        with open(os.path.join(out_dir, name + '.json'), 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=args.indent or None,
                      separators=(',', ':') if not args.indent else None)
        written += 1
        if written % 100 == 0:
            print('  %d / %d' % (written, len(names)), flush=True)

    print('Wrote %d %s sequences to %s' % (written, args.mode, out_dir))


if __name__ == '__main__':
    main()
