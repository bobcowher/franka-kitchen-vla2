#!/bin/bash
# Pull a run's checkpoint from Beekeeper into checkpoints/<run_ref>.pth.
# Usage: ./download.sh [run_id] [project]   (defaults: latest run, franka-kitchen-vla2)
set -euo pipefail

PROJECT=${2:-franka-kitchen-vla2}
API=${BEEKEEPER_URL:-http://lab.local:5000}/api/v1/projects/$PROJECT
REMOTE_FILE=checkpoints/smolvlm2.pth

json() { python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print($1)"; }

RUN=${1:-$(curl -sf "$API/runs" | json "max(r['run_id'] for r in d['runs'])")}
RUN_JSON=$(curl -sf "$API/runs/$RUN") || { echo "No run $RUN in $PROJECT" >&2; exit 1; }
read -r REF STATUS BRANCH < <(json "' '.join(d['run'][k] for k in ('run_ref', 'status', 'branch'))" <<< "$RUN_JSON")

OUT=checkpoints/$REF.pth
mkdir -p checkpoints
curl -sf "$API/runs/$RUN/files/$REMOTE_FILE" -o "$OUT.part"
mv "$OUT.part" "$OUT"

echo "$REF ($BRANCH, $STATUS) -> $OUT"
[ "$STATUS" = completed ] || echo "Warning: run is $STATUS, so this is its latest save, not the final one."
