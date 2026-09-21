using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace InterAct.Grab
{
    /// <summary>
    /// Deserialised form of a clip written by <c>export_grab_unity.py</c>.
    /// Positions and rotations are already in Unity's left-handed Y-up frame,
    /// so no axis conversion is needed here.
    /// </summary>
    [System.Serializable]
    public class GrabSkeleton
    {
        [JsonProperty("jointNames")] public List<string> JointNames;
        [JsonProperty("parents")] public List<int> Parents;

        /// <summary>Parent-relative rest offsets, one per joint.</summary>
        [JsonProperty("restLocalPositions")] public List<List<float>> RestLocalPositions;

        public Vector3 RestLocal(int joint)
        {
            var p = RestLocalPositions[joint];
            return new Vector3(p[0], p[1], p[2]);
        }
    }

    [System.Serializable]
    public class GrabObjectTrack
    {
        [JsonProperty("name")] public string Name;
        [JsonProperty("mesh")] public string Mesh;
        [JsonProperty("positions")] public List<List<float>> Positions;
        [JsonProperty("rotations")] public List<List<float>> Rotations;

        public Vector3 Position(int frame)
        {
            var p = Positions[frame];
            return new Vector3(p[0], p[1], p[2]);
        }

        public Quaternion Rotation(int frame)
        {
            var q = Rotations[frame];
            return new Quaternion(q[0], q[1], q[2], q[3]);
        }
    }

    /// <summary>One hand of a <c>hands</c>-mode clip, expressed in its own wrist frame.</summary>
    [System.Serializable]
    public class GrabHand
    {
        [JsonProperty("skeleton")] public GrabSkeleton Skeleton;
        [JsonProperty("localRotations")] public List<List<List<float>>> LocalRotations;
        [JsonProperty("object")] public GrabObjectTrack Object;

        public Quaternion LocalRotation(int frame, int joint)
        {
            var q = LocalRotations[frame][joint];
            return new Quaternion(q[0], q[1], q[2], q[3]);
        }
    }

    [System.Serializable]
    public class GrabClip
    {
        [JsonProperty("name")] public string Name;
        [JsonProperty("fps")] public int Fps;
        [JsonProperty("frameCount")] public int FrameCount;
        [JsonProperty("subject")] public string Subject;
        [JsonProperty("gender")] public string Gender;

        /// <summary>Either "fullbody" or "hands".</summary>
        [JsonProperty("mode")] public string Mode;

        [JsonProperty("text")] public string Text;

        // --- fullbody mode ---
        [JsonProperty("skeleton")] public GrabSkeleton Skeleton;
        [JsonProperty("rootPositions")] public List<List<float>> RootPositions;
        [JsonProperty("localRotations")] public List<List<List<float>>> LocalRotations;
        [JsonProperty("object")] public GrabObjectTrack Object;

        // --- hands mode ---
        [JsonProperty("hands")] public Dictionary<string, GrabHand> Hands;

        public bool IsHands => Mode == "hands";

        public Vector3 RootPosition(int frame)
        {
            var p = RootPositions[frame];
            return new Vector3(p[0], p[1], p[2]);
        }

        public Quaternion LocalRotation(int frame, int joint)
        {
            var q = LocalRotations[frame][joint];
            return new Quaternion(q[0], q[1], q[2], q[3]);
        }

        public static GrabClip Load(string path)
        {
            return JsonConvert.DeserializeObject<GrabClip>(File.ReadAllText(path));
        }

        public static GrabClip Parse(string json)
        {
            return JsonConvert.DeserializeObject<GrabClip>(json);
        }
    }
}
