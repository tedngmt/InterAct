using System.Collections.Generic;
using UnityEngine;

namespace InterAct.Grab
{
    /// <summary>
    /// Builds a joint hierarchy from a GRAB clip and plays it back.
    ///
    /// Drop this on an empty GameObject, point <see cref="clipPath"/> at one of
    /// the exported .json files and press play.  Works for both "fullbody" and
    /// "hands" clips; hand clips spawn one hierarchy per hand, each rooted at
    /// its wrist.
    /// </summary>
    public class GrabPlayer : MonoBehaviour
    {
        [Tooltip("Absolute path, or a path relative to the project folder.")]
        public string clipPath;

        [Tooltip("Optional prefab or mesh instance for the interacting object.")]
        public GameObject objectPrefab;

        public bool loop = true;
        public bool drawGizmos = true;

        GrabClip _clip;
        readonly List<Transform[]> _rigs = new List<Transform[]>();
        readonly List<GrabSkeleton> _skeletons = new List<GrabSkeleton>();
        readonly List<GrabHand> _hands = new List<GrabHand>();
        readonly List<Transform> _objects = new List<Transform>();

        float _time;

        void Start()
        {
            _clip = GrabClip.Load(clipPath);

            if (_clip.IsHands)
            {
                foreach (var kv in _clip.Hands)
                {
                    var root = new GameObject(kv.Key + "_wrist").transform;
                    root.SetParent(transform, false);
                    _rigs.Add(BuildRig(kv.Value.Skeleton, root));
                    _skeletons.Add(kv.Value.Skeleton);
                    _hands.Add(kv.Value);
                    _objects.Add(SpawnObject(kv.Value.Object, root));
                }
            }
            else
            {
                var root = new GameObject("root").transform;
                root.SetParent(transform, false);
                _rigs.Add(BuildRig(_clip.Skeleton, root));
                _skeletons.Add(_clip.Skeleton);
                _hands.Add(null);
                // The object moves in world space, so parent it to the clip, not the rig.
                _objects.Add(SpawnObject(_clip.Object, transform));
            }
        }

        Transform[] BuildRig(GrabSkeleton skel, Transform root)
        {
            var joints = new Transform[skel.JointNames.Count];
            for (int j = 0; j < joints.Length; j++)
            {
                var t = j == 0 ? root : new GameObject(skel.JointNames[j]).transform;
                if (j != 0)
                {
                    t.SetParent(joints[skel.Parents[j]], false);
                    t.localPosition = skel.RestLocal(j);
                }
                joints[j] = t;
            }
            return joints;
        }

        Transform SpawnObject(GrabObjectTrack track, Transform parent)
        {
            if (track == null) return null;
            var go = objectPrefab != null
                ? Instantiate(objectPrefab)
                : GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = track.Name;
            if (objectPrefab == null) go.transform.localScale = Vector3.one * 0.08f;
            go.transform.SetParent(parent, false);
            return go.transform;
        }

        void Update()
        {
            if (_clip == null || _clip.FrameCount == 0) return;

            _time += Time.deltaTime;
            float total = _clip.FrameCount / (float)_clip.Fps;
            if (_time > total)
            {
                if (!loop) { _time = total; }
                else { _time -= total; }
            }

            int frame = Mathf.Clamp(Mathf.FloorToInt(_time * _clip.Fps), 0, _clip.FrameCount - 1);
            Apply(frame);
        }

        void Apply(int frame)
        {
            for (int r = 0; r < _rigs.Count; r++)
            {
                var joints = _rigs[r];
                var hand = _hands[r];

                for (int j = 0; j < joints.Length; j++)
                {
                    joints[j].localRotation = hand != null
                        ? hand.LocalRotation(frame, j)
                        : _clip.LocalRotation(frame, j);
                }

                // Fullbody clips carry an explicit world-space root position.
                if (hand == null) joints[0].localPosition = _clip.RootPosition(frame);

                var obj = _objects[r];
                if (obj == null) continue;
                var track = hand != null ? hand.Object : _clip.Object;
                obj.localPosition = track.Position(frame);
                obj.localRotation = track.Rotation(frame);
            }
        }

        void OnDrawGizmos()
        {
            if (!drawGizmos || _rigs.Count == 0) return;
            Gizmos.color = Color.cyan;
            for (int r = 0; r < _rigs.Count; r++)
            {
                var joints = _rigs[r];
                var skel = _skeletons[r];
                for (int j = 1; j < joints.Length; j++)
                {
                    var p = joints[skel.Parents[j]];
                    if (p != null) Gizmos.DrawLine(p.position, joints[j].position);
                }
            }
        }
    }
}
