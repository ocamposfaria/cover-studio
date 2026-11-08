#!/bin/bash
# Compila o Doubler (Audio Unit v2) só com as Command Line Tools, sem Xcode.
#
#   ./build.sh            compila e testa em build/Doubler.component
#   ./build.sh install    idem, instala em ~/Library/Audio/Plug-Ins/Components
#                         e valida com auval
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SDK_TAG="AudioUnitSDK-1.4.0"
SDK="$HERE/.deps/AudioUnitSDK"
BUILD="$HERE/build"
BUNDLE="$BUILD/Doubler.component"
DEST="$HOME/Library/Audio/Plug-Ins/Components"

if [ ! -d "$SDK" ]; then
	echo "→ baixando o AudioUnitSDK da Apple ($SDK_TAG)"
	git clone -q -c advice.detachedHead=false --depth 1 --branch "$SDK_TAG" \
		https://github.com/apple/AudioUnitSDK "$SDK"
fi

ARCHS=(-arch arm64 -arch x86_64)
CXXFLAGS=(-std=c++23 -O2 -mmacosx-version-min=11.0 "${ARCHS[@]}" -I"$SDK/include" -Wall)
FRAMEWORKS=(-framework AudioToolbox -framework CoreAudio -framework CoreFoundation -framework CoreMIDI)

SDK_SRC=(AUBase AUBuffer AUBufferAllocator AUEffectBase AUInputElement AUOutputElement
	AUPlugInDispatch AUScopeElement ComponentBase AUMIDIBase MusicDeviceBase)
SOURCES=("$HERE/Doubler.cpp")
for f in "${SDK_SRC[@]}"; do SOURCES+=("$SDK/src/AudioUnitSDK/$f.cpp"); done

echo "→ compilando"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/Contents/MacOS"
clang++ "${CXXFLAGS[@]}" -bundle "${SOURCES[@]}" "${FRAMEWORKS[@]}" \
	-Wl,-exported_symbol,_DoublerFactory \
	-o "$BUNDLE/Contents/MacOS/Doubler"
cp "$HERE/Info.plist" "$BUNDLE/Contents/Info.plist"
printf 'BNDL????' > "$BUNDLE/Contents/PkgInfo"

# Assinatura ad-hoc: basta para carregar na própria máquina.
codesign --force --sign - --timestamp=none "$BUNDLE"

echo "→ testando o binário"
clang++ -std=c++23 -O1 "$HERE/test_render.cpp" -framework AudioToolbox -framework CoreFoundation \
	-o "$BUILD/test_render"
"$BUILD/test_render" "$BUNDLE/Contents/MacOS/Doubler"

if [ "${1:-}" = "install" ]; then
	echo "→ instalando em $DEST"
	mkdir -p "$DEST"
	rm -rf "$DEST/Doubler.component"
	cp -R "$BUNDLE" "$DEST/"
	# O registro de Audio Units fica em cache; derrubar o registrar força reler.
	killall -9 AudioComponentRegistrar 2>/dev/null || true
	echo "→ validando com auval"
	if auval -v aufx Dblr CvSt > "$BUILD/auval.log" 2>&1; then
		grep -E "AU VALIDATION SUCCEEDED" "$BUILD/auval.log"
	else
		tail -40 "$BUILD/auval.log"
		echo "✗ auval falhou (log completo em $BUILD/auval.log)"
		exit 1
	fi
	echo "✓ instalado. Reabra o GarageBand para ele achar o plugin."
fi
