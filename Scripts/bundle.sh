#!/usr/bin/env bash
#
# bundle.sh — Build Fever as a real macOS .app bundle, ad-hoc codesign it
# (with entitlements + hardened runtime), verify the signature, and install it
# into /Applications.
#
# Command-Line-Tools only (no Xcode/xcodebuild). Build via `swift build`.
#
set -euo pipefail

# --- Paths -------------------------------------------------------------------
# Derive the project root from this script's own location (Scripts/..), so the
# build works wherever the project lives — no hardcoded path to break on a move.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Fever"
BUNDLE_ID="com.fir4s.fever"   # MUST stay constant.
ENTITLEMENTS="${PROJECT_ROOT}/Fever/Fever.entitlements"

# IMPORTANT: the project lives on an iCloud/file-provider-synced volume
# (~/Documents) that asynchronously re-stamps com.apple.FinderInfo (and other)
# xattrs onto files. If we sign the bundle in place, the sync daemon can
# re-attach a FinderInfo xattr between `xattr -cr` and `codesign --verify`,
# which makes `codesign --verify --strict` reject the bundle as "detritus".
#
# Fix: assemble + sign + verify the bundle in a TEMP dir that is OFF the synced
# volume (/tmp is local, not file-provider-backed), then copy the already-signed
# bundle into /Applications and re-verify there.
BUILD_DIR="$(mktemp -d /tmp/fever-build.XXXXXX)"
APP_DIR="${BUILD_DIR}/${APP_NAME}.app"
CONTENTS="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS}/MacOS"
RESOURCES_DIR="${CONTENTS}/Resources"
INSTALLED_APP="/Applications/${APP_NAME}.app"

# Clean up the scratch build dir on exit (success or failure).
cleanup() { rm -rf "${BUILD_DIR}"; }
trap cleanup EXIT

cd "${PROJECT_ROOT}"

# --- 1. Build release --------------------------------------------------------
echo "==> Building ${APP_NAME} (release)…"
swift build -c release

# Resolve the actual product output directory.
BIN="$(swift build -c release --show-bin-path)"
echo "==> BIN: ${BIN}"

if [[ ! -x "${BIN}/${APP_NAME}" ]]; then
    echo "ERROR: built binary not found at ${BIN}/${APP_NAME}" >&2
    exit 1
fi

# --- 2. Assemble the .app skeleton (in OFF-volume temp dir) ------------------
echo "==> Assembling ${APP_NAME}.app in ${BUILD_DIR} (off synced volume)…"
rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# --- 3. Copy executable ------------------------------------------------------
cp "${BIN}/${APP_NAME}" "${MACOS_DIR}/${APP_NAME}"
chmod +x "${MACOS_DIR}/${APP_NAME}"

# --- 4. Write Info.plist -----------------------------------------------------
cat > "${CONTENTS}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Fever</string>
    <key>CFBundleIdentifier</key>
    <string>com.fir4s.fever</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleName</key>
    <string>Fever</string>
    <key>CFBundleDisplayName</key>
    <string>Fever</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>26.0</string>
    <key>NSCameraUsageDescription</key>
    <string>Fever uses the camera to track your full body and stream it to VRChat.</string>
    <key>NSLocalNetworkUsageDescription</key>
    <string>Fever sends VRChat trackers over OSC/UDP and connects to your tracking PC on your local network.</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.utilities</string>
</dict>
</plist>
PLIST

# --- 5. Write PkgInfo --------------------------------------------------------
printf 'APPL????' > "${CONTENTS}/PkgInfo"

# NOTE: Fever runs the NLF pose model through an EXTERNAL onnxruntime sidecar
# (resolved at runtime from $FEVER_NLF_ROOT, default ~/Dev/BodyPose3DDemo). The
# model is non-distributable, so nothing is embedded in the .app — there is no
# bundled Python/sidecar to stage or sign.

# --- 5b. Strip extended attributes -------------------------------------------
# Finder/Spotlight can attach a com.apple.FinderInfo (and other) xattrs to the
# bundle, which makes `codesign --verify --strict` reject it as "detritus".
# Clear them recursively before signing so verification stays reliable.
xattr -cr "${APP_DIR}"

# --- 6. Ad-hoc codesign (AFTER Info.plist), in the off-volume temp dir -------
echo "==> Codesigning (ad-hoc, hardened runtime, with entitlements) in ${BUILD_DIR}…"
codesign --force --options runtime \
    --entitlements "${ENTITLEMENTS}" \
    -s - "${APP_DIR}"

# --- 7. Verify signature in /tmp (must PASS before we install) ---------------
echo "==> codesign --verify --strict (temp/${APP_NAME}.app)"
codesign --verify --strict --verbose=2 "${APP_DIR}"

echo "==> codesign -dvvv (temp/${APP_NAME}.app)"
codesign -dvvv "${APP_DIR}"

# --- 8. Install the VERIFIED bundle into /Applications -----------------------
echo "==> Installing verified bundle into /Applications…"
rm -rf "${INSTALLED_APP}"
cp -R "${APP_DIR}" "${INSTALLED_APP}"

# Copying back onto a synced volume can re-attach FinderInfo xattrs. Strip them
# again on the installed copy. (The signature itself is unaffected — xattrs are
# not part of the signed payload — so this does not invalidate it.)
echo "==> Stripping xattrs on installed copy…"
xattr -cr "${INSTALLED_APP}"

# --- 9. Re-verify the /Applications copy ------------------------------------
echo "==> codesign --verify --strict (/Applications/${APP_NAME}.app)"
codesign --verify --strict --verbose=2 "${INSTALLED_APP}"

echo "==> Done. Built+signed in ${BUILD_DIR}, verified, installed to ${INSTALLED_APP}"
