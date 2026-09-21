"""Attach InterAct's GRAB text annotations to a converted SOMA-X GRAB package.

The SOMA package (``GRAB_SOMA_51Objects``) already carries the full motion for
all 1,335 GRAB sequences, but no natural-language descriptions — it only derives
a coarse ``motion_intent`` from the filename.  InterAct supplies three phrasings
per sequence (natural / change / shorten) plus an action label.

The annotation CSVs hold exactly one description set per whole GRAB sequence,
so this is a 1:1 join keyed on ``(subject, sequence)``.  (InterAct's own
pipeline additionally splits 10 of the 1,335 sequences into 2-3 sub-clips for
training; that segmentation does not apply here, since the SOMA package stores
whole sequences.)

Annotations are written as a **sidecar** tree next to ``motions/`` rather than
into the ``.npz`` files, so every motion file stays byte-identical and the
package's own ``reports/data_sha256.json`` validation keeps holding.

Usage::

    python soma_export/attach_grab_annotations.py \\
        --package ../GRAB_SOMA_51Objects

    # See what would happen without writing anything
    python soma_export/attach_grab_annotations.py --package ... --dry-run
"""

import argparse
import collections
import json
import os
import re
import sys

import pandas as pd

# The label vocabulary InterAct itself uses (process/process_text.py).
ACTION_SET = ['Rotate', 'Move', 'Carry', 'Hold', 'Play', 'Manipulate', 'Sit',
              'Bowling', 'Lift', 'Eat', 'Adjust', 'Swing', 'Pass', 'Exercise',
              'Kick', 'Drink']
_ACTION_LOOKUP = {a.lower(): a for a in ACTION_SET}

# Phrasing variants to carry across. 'raw' is the unedited annotator text.
VARIANTS = ['natural', 'change', 'shorten', 'raw']

URL_RE = re.compile(r'^GRAB_(?P<rest>.+)\.mp4$')


def normalise_action(value):
    """'Move.' / 'Adjust ' -> canonical ACTION_SET spelling, or None."""
    if value is None:
        return None
    cleaned = str(value).strip().rstrip('.').strip()
    return _ACTION_LOOKUP.get(cleaned.lower())


def parse_video_url(url, package_keys):
    """'GRAB_s1_apple_pass_1_apple.mp4' -> ('s1', 'apple_pass_1').

    The object name is the trailing token, but rather than trust that blindly
    we confirm the result against the sequences the package actually contains.
    """
    m = URL_RE.match(str(url).strip())
    if not m:
        return None
    parts = m.group('rest').split('_')
    if len(parts) < 3:
        return None
    subject = parts[0]
    sequence = '_'.join(parts[1:-1])   # drop the trailing object name
    if (subject, sequence) in package_keys:
        return subject, sequence
    # Fall back to the longest suffix split that the package recognises.
    for cut in range(2, len(parts)):
        cand = (parts[0], '_'.join(parts[1:cut]))
        if cand in package_keys:
            return cand
    return None


def load_package_keys(package):
    motions = os.path.join(package, 'motions')
    if not os.path.isdir(motions):
        sys.exit('No motions/ folder under %s — is that the SOMA package?' % package)
    keys = set()
    for subject in sorted(os.listdir(motions)):
        d = os.path.join(motions, subject)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith('.npz'):
                keys.add((subject, fn[:-4]))
    return keys


def load_annotations(annotation_root):
    """url -> {variant: [texts], 'action': label, 'behave': str}."""
    per_url = collections.defaultdict(lambda: {'texts': collections.defaultdict(list)})
    seen_variants = []

    for variant in VARIANTS:
        path = os.path.join(annotation_root, variant, 'grab.csv')
        if not os.path.exists(path):
            continue
        seen_variants.append(variant)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if str(row.get('AssignmentStatus', 'Approved')).strip() != 'Approved':
                continue
            text = str(row['Answer.action']).strip()
            if text and text.lower() != 'nan':
                per_url[row['Input.video_url']]['texts'][variant].append(text)
            behave = str(row.get('Answer.Behave', '')).strip()
            if behave and behave.lower() != 'nan':
                per_url[row['Input.video_url']]['behave'] = behave

    action_path = os.path.join(annotation_root, 'action', 'grab.csv')
    conflicts = []
    if os.path.exists(action_path):
        df = pd.read_csv(action_path)
        grouped = collections.defaultdict(set)
        for _, row in df.iterrows():
            label = normalise_action(row['Answer.action'])
            if label:
                grouped[row['Input.video_url']].add(label)
        for url, labels in grouped.items():
            if len(labels) > 1:
                conflicts.append((url, sorted(labels)))
            per_url[url]['action'] = sorted(labels)[0]

    return per_url, seen_variants, conflicts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--package', required=True,
                    help='path to the converted SOMA GRAB package')
    ap.add_argument('--annotations', default='./data/annotation',
                    help="InterAct's annotation folder (default: ./data/annotation)")
    ap.add_argument('--out', default=None,
                    help='sidecar output dir (default: <package>/annotations)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report the join without writing files')
    args = ap.parse_args()

    package_keys = load_package_keys(args.package)
    per_url, variants, conflicts = load_annotations(args.annotations)
    print('package sequences : %d' % len(package_keys))
    print('annotated urls    : %d  (variants: %s)' % (len(per_url), ', '.join(variants)))

    out_root = args.out or os.path.join(args.package, 'annotations')

    matched, unparsed, multi = {}, [], []
    for url, payload in per_url.items():
        key = parse_video_url(url, package_keys)
        if key is None:
            unparsed.append(url)
            continue
        texts = {v: payload['texts'].get(v, []) for v in variants}
        if any(len(t) > 1 for t in texts.values()):
            multi.append(url)
        matched[key] = {
            'subject': key[0],
            'sequence': key[1],
            'motion': 'motions/%s/%s.npz' % key,
            'action': payload.get('action'),
            'category': payload.get('behave'),
            # Lists, because a sequence may carry more than one annotator's text.
            'descriptions': {v: texts[v] for v in variants if texts[v]},
            'source': {'dataset': 'InterAct', 'video_url': url},
        }

    missing = sorted(package_keys - set(matched))
    print('matched           : %d' % len(matched))
    print('package seqs without annotation: %d%s'
          % (len(missing), (' -> %s' % missing[:3]) if missing else ''))
    print('urls not matched to package    : %d%s'
          % (len(unparsed), (' -> %s' % unparsed[:3]) if unparsed else ''))
    print('sequences with >1 annotator text: %d%s'
          % (len(multi), (' -> %s' % multi[:3]) if multi else ''))
    print('action-label conflicts          : %d%s'
          % (len(conflicts), (' -> %s' % conflicts[:3]) if conflicts else ''))

    no_action = [k for k, v in matched.items() if not v['action']]
    print('matched without action label    : %d' % len(no_action))

    if args.dry_run:
        print('\n--dry-run: nothing written.')
        return

    written = 0
    for (subject, sequence), payload in sorted(matched.items()):
        d = os.path.join(out_root, subject)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, sequence + '.json'), 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        written += 1

    manifest = {
        'source': 'InterAct GRAB text annotations',
        'variants': variants,
        'actionSet': ACTION_SET,
        'sequences': written,
        'packageSequences': len(package_keys),
        'unannotatedSequences': missing,
        'layout': 'annotations/<subject>/<sequence>.json, 1:1 with motions/<subject>/<sequence>.npz',
        'note': ('Each annotation covers one whole GRAB sequence: the CSVs hold a single '
                 'description set per sequence. InterAct splits 10 of the 1,335 sequences into '
                 'sub-clips for training, but that does not apply here because the SOMA package '
                 'stores whole sequences. Motion .npz files are untouched, so the package '
                 'SHA-256 reports remain valid.'),
    }
    with open(os.path.join(out_root, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print('\nWrote %d annotation files to %s' % (written, out_root))


if __name__ == '__main__':
    main()
