#!/usr/bin/env bash
# Render an HTML figure to a 2x PNG with headless Edge (Windows).
#   bash tools/render_figure.sh [HTML] [PNG] [HEIGHT_CSS_PX]
set -eu
html=${1:-assets/paper/architecture_overview.html}
png=${2:-assets/architecture-overview.png}
height=${3:-958}
edge="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
"$edge" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1600,"$height" \
  --virtual-time-budget=8000 --screenshot="$(cygpath -w "$PWD/$png")" "file:///$(cygpath -m "$PWD/$html")" 2>/dev/null
echo "wrote $png"
