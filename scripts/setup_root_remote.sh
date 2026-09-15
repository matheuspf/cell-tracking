#!/usr/bin/env bash
# Set up paths and shell entry points after copying the /root mirror.
# The instance disk is not an independently persistent Vast volume.
set -euo pipefail
test "$(id -u)" = 0 || { printf 'Run on the remote host as root.\n' >&2; exit 1; }
repo_path=/root/code/kaggle/cell-tracking
test -f "$repo_path/scripts/root_remote_env.sh"
mkdir -p /root/bin /root/tmp /root/setup/backups /root/.local/share/jupyter/kernels

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

link_path /home/mpf/code/kaggle/cell-tracking "$repo_path"
link_path /home/mpf/.conda /root/.conda
link_path /home/mpf/.codex /root/.codex
link_path /home/mpf/.agents /root/.agents
link_path /home/mpf/.cache /root/.cache
link_path /home/mpf/.local/share/jupyter /root/.local/share/jupyter
link_path /root/cell-tracking "$repo_path"
link_path /usr/lib/chatgpt/resources/cua_node /root/toolchain/cua_node
link_path /usr/lib/chatgpt/resources/codex /usr/local/bin/codex
for input_name in biohub-tracking-support-pack-50ep-v1 biohub-deepcenter-unet3d-center-prior-v1 biohub-temporal-unet3d-seed314159-v1; do
    link_path "/kaggle/input/$input_name" "/kaggle/input/datasets/pilkwang/$input_name"
done
link_path /kaggle/input/biohub-cell-tracking-during-development /kaggle/input/competitions/biohub-cell-tracking-during-development

env_line=". $repo_path/scripts/root_remote_env.sh"
for startup in /root/.bashrc /root/.profile /root/.zshrc /root/.zshenv; do
    touch "$startup"
    if ! grep -Fqx "$env_line" "$startup"; then
        printf '\n%s\n' "$env_line" >> "$startup"
    fi
done
# ssh command shells read .bashrc but return before its interactive section.
python3 - "$env_line" <<'PY'
from pathlib import Path
import sys
path = Path('/root/.bashrc')
line = sys.argv[1]
lines = [x for x in path.read_text().splitlines() if x != line]
path.write_text(line + '\n' + '\n'.join(lines) + '\n')
PY
printf '%s\n' "$env_line" > /etc/profile.d/cell-tracking.sh
cat > /root/bin/cell-tracking <<'SH'
#!/usr/bin/env bash
set -euo pipefail
. /root/code/kaggle/cell-tracking/scripts/root_remote_env.sh
conda activate cell-tracking
cd /root/code/kaggle/cell-tracking
if [ "$#" -gt 0 ]; then exec "$@"; fi
exec zsh -l
SH
chmod 755 /root/bin/cell-tracking
link_path /usr/local/bin/cell-tracking /root/bin/cell-tracking
touch /root/.tmux.conf
if ! grep -Fqx 'set -g mouse on' /root/.tmux.conf; then
    printf '\nset -g mouse on\n' >> /root/.tmux.conf
fi
if ! grep -Fqx 'set -g default-shell /bin/zsh' /root/.tmux.conf; then
    printf '\nset -g default-shell /bin/zsh\n' >> /root/.tmux.conf
fi
if tmux has-session 2>/dev/null; then
    tmux set -g mouse on
    tmux set -g default-shell /bin/zsh
fi
printf 'Ready: cell-tracking\n'
