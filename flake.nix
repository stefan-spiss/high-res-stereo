{
  description = "high-res-stereo — Python (torch wheels) + C++ (libtorch/OpenCV/TBB) dev environment, CUDA or CPU";

  inputs = {
    # Latest stable nixpkgs (26.05), pinned to a fixed rev in flake.lock. This is
    # the stable Linux channel (nixpkgs-26.05 exists only for darwin); the nixos-
    # prefix is just the channel name, not a NixOS-system dependency. opencv-cuda
    # and libtorch are prebuilt on cache.nixos-cuda.org for this rev.
    # Re-pin with:
    #   nix flake lock --override-input nixpkgs github:NixOS/nixpkgs/<rev>
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { self, nixpkgs, flake-utils }:
    let
      inherit (nixpkgs) lib;

      # Builds a dev shell with or without CUDA.
      # Nix can't probe the host GPU at eval time, so we expose both as separate
      # shells and let the user pick. cudaSupport toggles the whole stack in one
      # place: opencv's CUDA modules, the libtorch-bin flavour (+cu130 vs. cpu),
      # the CUDA-13 companions (cudatoolkit/cuDNN/NVRTC) and the PyTorch wheel
      # index (cu130 vs. cpu) used for the Python .venv.
      #
      # On macOS, CUDA is unavailable; cudaSupport is silently overridden to false.
      # The darwin libtorch-bin wheel already
      # includes the MPS backend, so torch.device('mps') works out of the box
      # on Apple Silicon with no extra packages — Metal is a system framework
      # found automatically at /System/Library/Frameworks/.
      mkDevShell =
        { system, cudaSupport }:
        let
          isDarwin = lib.hasSuffix "darwin" system;

          # macOS has no NVIDIA CUDA; ignore whatever the caller passed.
          effectiveCudaSupport = if isDarwin then false else cudaSupport;

          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = effectiveCudaSupport;
            };
            # On macOS, libtorch-bin's libtorch_cpu.dylib hard-codes a Homebrew
            # absolute path for libomp even though Homebrew may not be present.
            # Redirect it to Nix's libomp (the same one OpenCV links) so both
            # share a single OpenMP runtime — no dual-libomp corruption.
            #
            # Critical: do NOT call codesign after install_name_tool.
            # On macOS 16+, install_name_tool auto-re-signs and preserves the
            # linker-signed flag that Darwin 25 requires (flags=0x20002).
            # An explicit `codesign --sign - --force` strips that flag
            # (flags=0x2 only) and the process is killed at dylib load time.
            overlays = lib.optionals isDarwin [
              (final: prev: {
                libtorch-bin = prev.libtorch-bin.overrideAttrs (old: {
                  postFixup = (old.postFixup or "") + ''
                    /usr/bin/install_name_tool \
                      -change "/opt/homebrew/opt/libomp/lib/libomp.dylib" \
                              "${final.llvmPackages.openmp}/lib/libomp.dylib" \
                      "$out/lib/libtorch_cpu.dylib"
                  '';
                });
              })
            ];
          };

          python = pkgs.python313;

          # .venv label and shell name:
          #   cuda        — Linux + NVIDIA GPU
          #   mps         — Apple Silicon (MPS backend is in the standard macOS wheel)
          #   cpu         — Linux CPU-only or Intel Mac
          accel =
            if effectiveCudaSupport then
              "cuda"
            else if system == "aarch64-darwin" then
              "mps"
            else
              "cpu";

          # CUDA package set: CUDA 13, to match the prebuilt PyTorch used here.
          # torch 2.9.0 is distributed built against CUDA 13 (Python wheel cu130 +
          # the +cu130 libtorch-bin), and cu13 supports the RTX 5070 (Blackwell,
          # sm_120). PyTorch ships no cu12.9 build, so nixpkgs' default
          # cudaPackages (12.9) can't match -> pin cudaPackages_13. Guarded
          # behind effectiveCudaSupport to avoid evaluation on macOS where
          # cudaPackages is not defined.
          cuda = if effectiveCudaSupport then pkgs.cudaPackages_13 else { };

          # ===================================================================
          # C++ deps. opencv's CUDA modules and the libtorch-bin flavour both
          # follow effectiveCudaSupport (CPU builds when false).
          # ===================================================================
          opencv = pkgs.opencv4.override {
            enableCuda = effectiveCudaSupport;
            enableContrib = true; # aruco/charuco etc.
            enableGtk3 = !isDarwin; # macOS uses Cocoa/AppKit (auto-enabled by nixpkgs)
            enableFfmpeg = true;
          };

          # Prebuilt libtorch binary. cudaSupport picks the flavour: +cu130 for GPU,
          # cpu/mps for macOS and CPU Linux. On macOS, pkgs.libtorch-bin resolves
          # through the overlay that redirects the Homebrew libomp reference to
          # Nix's libomp — ensuring a single OpenMP runtime shared with OpenCV.
          libtorchBin = pkgs.libtorch-bin;

          # PyTorch wheel index for the Python .venv: cu130 for GPU, cpu otherwise.
          # pytorch.org/whl/cpu serves both Linux and macOS wheels; pip selects the
          # correct platform tag automatically. PyPI is the extra-index fallback.
          torchIndexUrl =
            if effectiveCudaSupport then
              "https://download.pytorch.org/whl/cu130"
            else
              "https://download.pytorch.org/whl/cpu";

          # ===================================================================
          # Python — pure prebuilt wheels in a Nix-managed .venv. Nothing here
          # compiles; only downloads.
          # ===================================================================

          # sha256sum is GNU coreutils (Linux); shasum -a 256 is the macOS BSD util.
          sha256Cmd = if isDarwin then "shasum -a 256" else "sha256sum";

          # Runtime libs the wheels dlopen:
          #   Linux  — libstdc++, zlib, and the full xcb/GL/glib stack for
          #            opencv-python's bundled Qt GUI (cv2.imshow via xcb).
          #   Darwin — libstdc++ and zlib only; the macOS opencv-python wheel
          #            bundles its own Qt (macOS platform plugin). libomp is
          #            handled via the nixpkgs overlay (Nix abs path in libtorch).
          pythonRuntimeLibs =
            if isDarwin then
              lib.makeLibraryPath (
                with pkgs;
                [
                  stdenv.cc.cc.lib
                  zlib
                ]
              )
            else
              lib.makeLibraryPath (
                with pkgs;
                [
                  stdenv.cc.cc.lib
                  zlib
                  # OpenCV-python Qt GUI (xcb platform plugin) dependencies:
                  libGL
                  glib
                  fontconfig.lib
                  freetype
                  libxkbcommon
                  dbus.lib
                  libx11
                  libxext
                  libxrender
                  libxcb
                  libxau
                  libxdmcp
                  libsm
                  libice
                  libxcb-util
                  libxcb-image
                  libxcb-keysyms
                  libxcb-render-util
                  libxcb-wm
                  libxcb-cursor
                ]
              );

          shellHook = ''
            # ---------- C++ ----------
            export Torch_DIR=${libtorchBin.dev}/share/cmake/Torch
          ''
          + lib.optionalString effectiveCudaSupport ''
            export CUDA_PATH=${cuda.cudatoolkit}
            # nvcc rejects our newer gcc/clang; pin the CUDA host compiler.
            export CUDAHOSTCXX=${cuda.backendStdenv.cc}/bin/g++
          ''
          + ''
            # ---------- Python: wheel-based .venv (first run downloads wheels) ----
            # A per-accel venv (.venv-cuda / .venv-mps / .venv-cpu) so switching
            # shells doesn't force a multi-GB torch reinstall each time. The stamp
            # keys on the requirements file *and* the accel variant, so editing the
            # requirements (or switching the torch index) re-installs, and an
            # interrupted install is retried.
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            _venv=.venv-${accel}
            if [ ! -e "$_venv/bin/python" ]; then
              echo "creating $_venv ..."
              ${python}/bin/python -m venv "$_venv"
            fi
            _want=$(${sha256Cmd} requirements-wheels.txt | cut -d' ' -f1)-${accel}
            if [ "$(cat "$_venv/.requirements.sha256" 2>/dev/null)" != "$_want" ]; then
              echo "installing torch (${accel}) + deps as wheels (downloads several GB the first time)..."
              "$_venv/bin/python" -m pip install --upgrade pip \
                && "$_venv/bin/python" -m pip install \
                     --index-url ${torchIndexUrl} \
                     --extra-index-url https://pypi.org/simple \
                     -r requirements-wheels.txt \
                && echo "$_want" > "$_venv/.requirements.sha256" \
                || echo "!! wheel install failed — re-enter the shell to retry"
            fi
            source "$_venv/bin/activate"
          ''
          + lib.optionalString (!isDarwin) ''
            # opencv-python's bundled Qt only ships the xcb platform plugin (no
            # wayland), so cv2.imshow aborts on a Wayland session. Force xcb -> runs
            # via XWayland (xcb runtime libs are on LD_LIBRARY_PATH below).
            export QT_QPA_PLATFORM=xcb
          ''
          + (
            if isDarwin then
              ''
                # ---------- Loader path (macOS) ----------
                export DYLD_LIBRARY_PATH=${pythonRuntimeLibs}''${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
              ''
            else if effectiveCudaSupport then
              ''
                # ---------- Loader path: cuDNN + NVRTC (for libtorch) + wheel/Qt libs ----------
                # libtorch-bin/torch ship without cuDNN and dlopen it (conv2d etc.) at
                # runtime, and dlopen NVRTC for their fuser -> provide both from the
                # CUDA-13 set to match the cu13 torch.
                export LD_LIBRARY_PATH=${cuda.cudnn.lib}/lib:${cuda.cuda_nvrtc.lib}/lib:${pythonRuntimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
              ''
            else
              ''
                # ---------- Loader path: wheel/Qt runtime libs (no CUDA) ----------
                export LD_LIBRARY_PATH=${pythonRuntimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
              ''
          )
          + ''
            echo "high-res-stereo dev shell (${accel})"
            echo "  Python : $(python --version 2>&1)  (torch 2.9.0 ${accel} wheel${
              if isDarwin then "" else ", cv2 Qt GUI"
            })"
          ''
          + ''
            echo "  C++    : OpenCV $(pkg-config --modversion opencv4 2>/dev/null || echo 4.x), libtorch ${libtorchBin.version} (${accel}), TBB"
          ''
          + lib.optionalString effectiveCudaSupport ''
            echo "  cuDNN ${cuda.cudnn.version} | NVRTC ${cuda.cuda_nvrtc.version} | CUDAHOSTCXX set"
          ''
          + lib.optionalString (system == "aarch64-darwin") ''
            echo "  MPS backend available — use torch.device('mps') for GPU tensors"
          ''
          + ''
            echo "  Torch_DIR set"
            echo "  Build C++:  rm -rf build && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -C build"
          '';
        in
        pkgs.mkShell {
          name = "high-res-stereo-dev-${accel}";

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            gcc
            python # interpreter for the venv
          ];

          buildInputs =
            with pkgs;
            [
              opencv
              glm
              tbb
              llvmPackages.openmp
              libtorchBin
            ]
            ++ lib.optionals effectiveCudaSupport [
              cuda.cudatoolkit
              cuda.cudnn
            ];

          inherit shellHook;
        };
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        isDarwin = lib.hasSuffix "darwin" system;
      in
      {
        devShells = {
          # CUDA/GPU shell on Linux (default); darwin silently uses CPU/MPS.
          #   nix develop
          default = mkDevShell {
            inherit system;
            cudaSupport = system == "x86_64-linux";
          };
        }
        // lib.optionalAttrs (!isDarwin) {
          # CPU-only shell for Linux machines without an NVIDIA GPU:
          #   nix develop .#cpu
          cpu = mkDevShell {
            inherit system;
            cudaSupport = false;
          };
        };
      }
    );
}
