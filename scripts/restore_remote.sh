#!/usr/bin/env bash
# Reattach an existing mirror after replacing the ephemeral Ubuntu container.
# This never downloads competition data or reinstalls the mirrored Python envs.
set -euo pipefail
test "$(id -u)" = 0 || { printf 'Run as root on the remote container.\n' >&2; exit 1; }
test -d /workspace/code/kaggle/cell-tracking
mkdir -p /workspace/setup/backups /workspace/setup/logs /workspace/bin \
    /workspace/tmp/jupyter /workspace/cache /workspace/conda/envs \
    /workspace/home/root/.config /workspace/home/root/.local/share \
    /workspace/home/root/.local/state /workspace/home/root/.jupyter
export TMPDIR=/workspace/tmp

if ! command -v rsync >/dev/null || ! command -v zsh >/dev/null || ! command -v rg >/dev/null || ! command -v vim >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends rsync zsh ripgrep vim \
        ca-certificates curl bzip2 unzip git tmux \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2 \
        libpango-1.0-0 libcairo2 libglib2.0-0 libdbus-1-3 libx11-6 libxcb1 \
        libxext6 libxcursor1 libxi6 libxtst6 fonts-liberation
fi

link_path() {
    local alias_path=$1 target_path=$2
    mkdir -p "$(dirname "$alias_path")"
    if [ -L "$alias_path" ] && [ "$(readlink -f "$alias_path")" = "$(readlink -f "$target_path")" ]; then
        return
    fi
    if [ -e "$alias_path" ] || [ -L "$alias_path" ]; then
        printf 'Refusing to replace existing path: %s\n' "$alias_path" >&2
        return 1
    fi
    ln -s "$target_path" "$alias_path"
}

link_path /kaggle /workspace/kaggle
link_path /home/mpf/code/kaggle/cell-tracking /workspace/code/kaggle/cell-tracking
link_path /home/mpf/.conda /workspace/conda
if [ -d /root/.conda ] && [ ! -L /root/.conda ]; then
    if [ -d /root/.conda/envs ] || [ -d /root/.conda/pkgs ]; then
        printf 'Existing Conda environments need an explicit migration: /root/.conda\n' >&2
        exit 1
    fi
    rsync -a /root/.conda/ /workspace/conda/
    mv /root/.conda "/workspace/setup/backups/conda-registry-$(date -u +%Y%m%dT%H%M%SZ)"
fi
link_path /root/.conda /workspace/conda
link_path /home/mpf/.codex /workspace/home/root/.codex
link_path /home/mpf/.agents /workspace/home/root/.agents
link_path /root/.agents /workspace/home/root/.agents
link_path /home/mpf/.cache /workspace/cache
if [ -d /workspace/toolchain/cua_node ]; then
    link_path /usr/lib/chatgpt/resources/cua_node /workspace/toolchain/cua_node
    link_path /usr/lib/chatgpt/resources/codex /workspace/bin/codex
fi
for input_name in biohub-tracking-support-pack-50ep-v1 biohub-deepcenter-unet3d-center-prior-v1 biohub-temporal-unet3d-seed314159-v1; do
    link_path "/workspace/kaggle/input/$input_name" "/workspace/kaggle/input/datasets/pilkwang/$input_name"
done
link_path /workspace/kaggle/input/biohub-cell-tracking-during-development /workspace/kaggle/input/competitions/biohub-cell-tracking-during-development

# The initial migration preserves the remote login before creating this alias.
if [ -d /workspace/home/root/.codex ]; then
    if [ -d /root/.codex ] && [ ! -L /root/.codex ]; then
        backup=/workspace/setup/backups/codex-before-restore-$(date -u +%Y%m%dT%H%M%SZ)
        cp -a /root/.codex "$backup"
        if [ -f /root/.codex/auth.json ]; then
            install -m 600 /root/.codex/auth.json /workspace/home/root/.codex/auth.json
        fi
        mv /root/.codex "/root/.codex.before-restore-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    link_path /root/.codex /workspace/home/root/.codex
fi
if [ -d /workspace/home/root/.kaggle ]; then
    link_path /root/.kaggle /workspace/home/root/.kaggle
fi
if [ -f /workspace/home/root/.gitconfig ]; then
    link_path /root/.gitconfig /workspace/home/root/.gitconfig
fi
# An already-running Jupyter server discovers these same persistent kernels.
if [ -d /root/.local/share/jupyter ] && [ ! -L /root/.local/share/jupyter ]; then
    rsync -a --ignore-existing /root/.local/share/jupyter/ /workspace/home/root/.local/share/jupyter/
    mv /root/.local/share/jupyter "/workspace/setup/backups/jupyter-data-$(date -u +%Y%m%dT%H%M%SZ)"
fi
link_path /root/.local/share/jupyter /workspace/home/root/.local/share/jupyter

repo_path=/workspace/code/kaggle/cell-tracking
env_line=". $repo_path/scripts/remote_env.sh"
for startup in /root/.bashrc /root/.profile /root/.zshrc /root/.zshenv; do
    touch "$startup"
    if ! grep -Fqx "$env_line" "$startup"; then
        printf '\n%s\n' "$env_line" >> "$startup"
    fi
done
printf '%s\n' "$env_line" > /etc/profile.d/cell-tracking.sh

cat > /workspace/bin/cell-tracking <<'SH'
#!/usr/bin/env bash
set -euo pipefail
. /workspace/code/kaggle/cell-tracking/scripts/remote_env.sh
cd /workspace/code/kaggle/cell-tracking
if [ "$#" -gt 0 ]; then exec "$@"; fi
exec zsh -l
SH
chmod 755 /workspace/bin/cell-tracking
link_path /usr/local/bin/cell-tracking /workspace/bin/cell-tracking
cat > /workspace/restore.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
exec bash /workspace/code/kaggle/cell-tracking/scripts/restore_remote.sh "$@"
SH
chmod 755 /workspace/restore.sh
. "$repo_path/scripts/remote_env.sh"
printf 'Persistent workspace restored. Run: cell-tracking\n'
