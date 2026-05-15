#!/usr/bin/env bash
# Create a dedicated conda env for pdf2ppt --inpaint-engine lama-pytorch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${LAMA_CONDA_ENV:-lama}"
PYTHON_VERSION="${LAMA_PYTHON_VERSION:-3.10}"

if [[ ! -x "$CONDA" ]]; then
  echo "conda not found at $CONDA; set CONDA to your conda binary." >&2
  exit 1
fi

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA" create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

LAMA_PY="$("$CONDA" info --base)/envs/${ENV_NAME}/bin/python"
"$LAMA_PY" -m pip install --upgrade pip wheel
"$LAMA_PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
"$LAMA_PY" -m pip install "numpy<2" "setuptools<81"
"$LAMA_PY" -m pip install \
  pyyaml tqdm easydict==1.9.0 \
  scikit-learn scikit-image opencv-python-headless \
  joblib matplotlib pandas packaging tabulate \
  hydra-core==1.1.0 omegaconf pytorch-lightning==1.2.9 \
  kornia==0.5.0 albumentations==0.5.2 webdataset

if [[ -d "${REPO_ROOT}/lama/.git" ]]; then
  bash "${REPO_ROOT}/scripts/apply_lama_patches.sh"
else
  echo "Skipping LaMa patches: clone https://github.com/advimman/lama into ${REPO_ROOT}/lama first." >&2
fi

export PYTHONPATH="${REPO_ROOT}/lama${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${REPO_ROOT}/lama"
"$LAMA_PY" -c "import saicinpainting.evaluation.utils; import torch; print('lama env ready:', torch.__version__, 'cuda=', torch.cuda.is_available())"

echo ""
echo "Conda env '${ENV_NAME}' is ready."
echo "Use with pdf2ppt:"
echo "  export PDF2PPT_LAMA_PYTHON=$LAMA_PY"
echo "  pdf2ppt input.pdf output.pptx --inpaint-engine lama-pytorch \\"
echo "    --inpaint-model-root lama/big-lama --inpaint-lama-repo-root lama"
