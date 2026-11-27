#!/usr/bin/env bash
# Turns a built site into redirect stubs pointing at the new domain.
# Used only by the GitHub Pages workflow, after agastyapatri.com is live,
# so old agastyapatri.github.io links land on the same page at the new address.
set -euo pipefail
dir="${1:-public}"
target="${2:-https://agastyapatri.com}"
find "$dir" -name 'index.html' | while read -r f; do
  path="${f#"$dir"}"; path="${path%index.html}"
  url="${target}${path}"
  cat > "$f" <<HTML
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Moved to ${url}</title>
<link rel="canonical" href="${url}">
<meta name="robots" content="noindex">
<meta http-equiv="refresh" content="0; url=${url}">
<script>location.replace("${url}" + location.hash)</script>
</head><body><p>This site has moved to <a href="${url}">${url}</a>.</p></body></html>
HTML
done
# Remove the old feeds so readers stop polling a copy that no longer updates.
find "$dir" -name 'index.xml' -delete
echo "Wrote redirect stubs in $dir -> $target"
