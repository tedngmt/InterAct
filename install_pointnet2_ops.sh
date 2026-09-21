#!/bin/bash
# Build pointnet2_ops against the CUDA toolkit that matches the installed torch.
#
# pointnet2_ops compiles CUDA kernels, so it needs nvcc.  Rather than assume a
# system CUDA install at a fixed path, this resolves CUDA_HOME in order:
#   1. $CUDA_HOME, if already set and usable
#   2. a sibling conda env holding a toolkit (default: cuda118)
#   3. /usr/local/cuda
#
# Override any of these from the environment:
#   CONDA_ENV      target env to install into        (default: interact)
#   CUDA_ENV       conda env providing nvcc          (default: cuda118)
#   CUDA_HOME      explicit toolkit root             (default: auto-detected)
#   TORCH_CUDA_ARCH_LIST  GPU architectures to build (default: 8.6, RTX 30xx)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/_pointnet2_build"
CONDA_ENV="${CONDA_ENV:-interact}"
CUDA_ENV="${CUDA_ENV:-cuda118}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

CONDA_BASE="$(conda info --base)"

if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  :
elif [ -x "$CONDA_BASE/envs/$CUDA_ENV/bin/nvcc" ]; then
  CUDA_HOME="$CONDA_BASE/envs/$CUDA_ENV"
elif [ -x "/usr/local/cuda/bin/nvcc" ]; then
  CUDA_HOME="/usr/local/cuda"
else
  echo "No nvcc found. Create a toolkit env matching your torch build, e.g.:" >&2
  echo "  conda create -y -n $CUDA_ENV -c nvidia/label/cuda-11.8.0 \\" >&2
  echo "      cuda-nvcc cuda-cudart-dev cuda-cccl" >&2
  exit 1
fi

NVCC_VER="$("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
TORCH_CUDA="$(conda run -n "$CONDA_ENV" python -c 'import torch; print(torch.version.cuda)')"
echo "CUDA_HOME : $CUDA_HOME (nvcc $NVCC_VER)"
echo "torch     : built for CUDA $TORCH_CUDA"
echo "arch list : $TORCH_CUDA_ARCH_LIST"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib

# The upstream setup.py pins its own arch list; replace it with ours.
sed -i "s/os.environ\[\"TORCH_CUDA_ARCH_LIST\"\] = .*/os.environ[\"TORCH_CUDA_ARCH_LIST\"] = \"$TORCH_CUDA_ARCH_LIST\"/" setup.py

# torch refuses to build when its CUDA major.minor differs from nvcc's. That
# check is only bypassed when the versions genuinely differ; a matching pair
# (as with torch cu118 + nvcc 11.8) compiles without patching.
if [ "${NVCC_VER%.*}" != "${TORCH_CUDA%.*}" ]; then
  echo "nvcc/torch CUDA mismatch - disabling torch's version check"
  sed -i '/from torch.utils.cpp_extension import/i \
import torch.utils.cpp_extension\ntorch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None' setup.py
fi

CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH" \
  conda run -n "$CONDA_ENV" --no-capture-output pip install .

cd "$SCRIPT_DIR"
rm -rf "$WORK_DIR"

conda run -n "$CONDA_ENV" python -c \
  "import pointnet2_ops, torch; from pointnet2_ops import pointnet2_utils; print('pointnet2_ops OK')"

echo "Done."
