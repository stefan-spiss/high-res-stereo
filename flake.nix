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
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      # Builds a dev shell either with CUDA (GPU) or as a CPU-only variant.
      # Nix can't probe the host GPU at eval time, so we expose both as separate
      # shells and let the user pick. cudaSupport toggles the whole stack in one
      # place: opencv's CUDA modules, the libtorch-bin flavour (+cu130 vs. cpu),
      # the CUDA-13 companions (cudatoolkit/cuDNN/NVRTC) and the PyTorch wheel
      # index (cu130 vs. cpu) used for the Python .venv.
      mkDevShell = { cudaSupport }:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              inherit cudaSupport;
            };
          };

          lib = pkgs.lib;
          python = pkgs.python313;
          accel = if cudaSupport then "cuda" else "cpu";

          # CUDA package set: CUDA 13, to match the prebuilt PyTorch used here.
          # torch 2.9.0 is distributed built against CUDA 13 (Python wheel cu130 +
          # the +cu130 libtorch-bin), and cu13 supports the RTX 5070 (Blackwell,
          # sm_120). PyTorch ships no cu12.9 build, so nixpkgs' default
          # cudaPackages (12.9) can't match -> pin cudaPackages_13. Only forced
          # by the CUDA variant (lazy: unused in the CPU shell).
          cuda = pkgs.cudaPackages_13;

          # ===================================================================
          # C++ dependencies. opencv's CUDA modules and the libtorch-bin flavour
          # both follow `cudaSupport` above (CPU builds when false).
          # ===================================================================
          opencv = pkgs.opencv4.override {
            enableCuda = cudaSupport;
            enableContrib = true;   # aruco/charuco etc.
            enableGtk3 = true;      # cv::imshow window support
            enableFfmpeg = true;
          };
          libtorchBin = pkgs.libtorch-bin;  # +cu130 (CUDA) or cpu build, prebuilt

          # PyTorch wheel index for the Python .venv: cu130 wheels for the GPU
          # variant, CPU-only wheels otherwise. torch/torchvision versions and the
          # rest of the deps live in requirements-wheels.txt.
          torchIndexUrl =
            if cudaSupport
            then "https://download.pytorch.org/whl/cu130"
            else "https://download.pytorch.org/whl/cpu";

          # ===================================================================
          # Python — pure prebuilt wheels in a Nix-managed .venv. Nothing here
          # compiles; only downloads.
          # ===================================================================

          # Runtime libs the wheels dlopen. torch needs libstdc++; opencv-python's
          # Qt GUI (cv2.imshow) needs the xcb/GL/glib stack; NixOS GPU driver lives
          # in /run/opengl-driver/lib.
          pythonRuntimeLibs = lib.makeLibraryPath (with pkgs; [
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
          ]);

          shellHook = ''
            # ---------- C++ ----------
            export Torch_DIR=${libtorchBin.dev}/share/cmake/Torch
          '' + lib.optionalString cudaSupport ''
            export CUDA_PATH=${cuda.cudatoolkit}
            # nvcc rejects our newer gcc/clang; pin the CUDA host compiler.
            export CUDAHOSTCXX=${cuda.backendStdenv.cc}/bin/g++
          '' + ''
            # ---------- Python: wheel-based .venv (first run downloads wheels) ----
            # A per-accel venv (.venv-cuda / .venv-cpu) so switching shells doesn't
            # force a multi-GB torch reinstall each time. The stamp keys on the
            # requirements file *and* the accel, so editing the requirements (or
            # switching the torch index) re-installs, and an interrupted install
            # is retried.
            export PIP_DISABLE_PIP_VERSION_CHECK=1
            _venv=.venv-${accel}
            if [ ! -e "$_venv/bin/python" ]; then
              echo "creating $_venv ..."
              ${python}/bin/python -m venv "$_venv"
            fi
            _want=$(sha256sum requirements-wheels.txt | cut -d' ' -f1)-${accel}
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

            # opencv-python's bundled Qt only ships the xcb platform plugin (no
            # wayland), so cv2.imshow aborts on a Wayland session. Force xcb -> runs
            # via XWayland (xcb runtime libs are on LD_LIBRARY_PATH below).
            export QT_QPA_PLATFORM=xcb
          '' + (if cudaSupport then ''
            # ---------- Loader path: cuDNN + NVRTC (for libtorch) + wheel/Qt libs ----------
            # libtorch-bin/torch ship without cuDNN and dlopen it (conv2d etc.) at
            # runtime, and dlopen NVRTC for their fuser -> provide both from the
            # CUDA-13 set to match the cu13 torch.
            export LD_LIBRARY_PATH=${cuda.cudnn.lib}/lib:${cuda.cuda_nvrtc.lib}/lib:${pythonRuntimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '' else ''
            # ---------- Loader path: wheel/Qt runtime libs (no CUDA) ----------
            export LD_LIBRARY_PATH=${pythonRuntimeLibs}:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          '') + ''
            echo "high-res-stereo dev shell (${accel})"
            echo "  Python : $(python --version 2>&1)  (torch 2.9.0 ${accel} wheel, cv2 Qt GUI)"
            echo "  C++    : OpenCV $(pkg-config --modversion opencv4 2>/dev/null || echo 4.x) (${accel}), libtorch ${libtorchBin.version} (${accel}), TBB"
          '' + lib.optionalString cudaSupport ''
            echo "  cuDNN ${cuda.cudnn.version} | NVRTC ${cuda.cuda_nvrtc.version} | CUDAHOSTCXX set"
          '' + ''
            echo "  Torch_DIR set"
            echo "  Build C++:  rm -rf build && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release && ninja -C build"
          '';
        in pkgs.mkShell {
          name = "high-res-stereo-dev-${accel}";

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            pkg-config
            gcc
            python           # interpreter for the venv
          ];

          buildInputs = with pkgs; [
            # C++ deps (prebuilt on cache.nixos-cuda.org for the pinned rev)
            opencv
            glm
            tbb
            llvmPackages.openmp
            libtorchBin
          ] ++ lib.optionals cudaSupport [
            cuda.cudatoolkit
            cuda.cudnn
          ];

          inherit shellHook;
        };
    in {
      devShells.${system} = {
        # CUDA/GPU shell (default):  nix develop
        default = mkDevShell { cudaSupport = true; };
        # CPU-only shell for machines without an NVIDIA GPU:  nix develop .#cpu
        cpu = mkDevShell { cudaSupport = false; };
      };
    };
}
