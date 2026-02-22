#!/bin/bash
# Run on server 89.218.178.215 as administrator (or root then fix ownership).
# One-time setup: install dependencies, clone repo, nginx, SSL.

set -e
SERVER_USER="${SERVER_USER:-administrator}"
APP_DIR="/var/www/zhan-ai"
REPO_URL="${REPO_URL:-}"  # optional: set your repo URL; or upload project via rsync/scp to /var/www/zhan-ai

echo "=== ZhanAI server setup ==="

# Optional: run as root to install system packages
if command -v apt-get &>/dev/null; then
  echo "Installing system packages (may need sudo)..."
  sudo apt-get update
  sudo apt-get install -y nginx certbot python3-certbot-nginx git python3-venv python3-pip nodejs npm
elif command -v yum &>/dev/null; then
  echo "Installing system packages (may need sudo)..."
  sudo yum install -y nginx certbot python3-certbot-nginx git python3
  # Node: consider installing from NodeSource or nvm
fi

# App directory
sudo mkdir -p "$APP_DIR"
sudo chown "$SERVER_USER:$SERVER_USER" "$APP_DIR"

# If you use git (set REPO_URL and have key/auth):
# cd /tmp && git clone "$REPO_URL" repo && cp -a repo/. "$APP_DIR/" && rm -rf repo

# Or: you will upload project via rsync/scp to $APP_DIR (see README)

# Python venv and backend deps
cd "$APP_DIR"
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# Frontend: install and build (after files are in place)
if [[ -d "$APP_DIR/frontend" ]]; then
  cd "$APP_DIR/frontend"
  npm ci
  echo "NEXT_PUBLIC_API_URL=https://api.zhan-ai.kz/predict/" > .env.production
  npm run build
else
  echo "Skip frontend build: $APP_DIR/frontend not found. Upload project and run deploy.sh"
fi

# Checkpoint: backend needs checkpoints/odir_best.pt
if [[ ! -f "$APP_DIR/checkpoints/odir_best.pt" ]]; then
  echo "WARNING: Put checkpoints/odir_best.pt into $APP_DIR/checkpoints/"
fi

# Nginx: install site config
sudo cp "$APP_DIR/deploy/nginx-zhan-ai.conf" /etc/nginx/sites-available/zhan-ai.conf
sudo ln -sf /etc/nginx/sites-available/zhan-ai.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# SSL (after DNS for zhan-ai.kz and api.zhan-ai.kz points to 89.218.178.215)
echo "Requesting SSL certificate..."
sudo certbot --nginx -d zhan-ai.kz -d www.zhan-ai.kz -d api.zhan-ai.kz --non-interactive --agree-tos -m admin@zhan-ai.kz || true

# Systemd services
sudo cp "$APP_DIR/deploy/zhan-api.service" /etc/systemd/system/
sudo cp "$APP_DIR/deploy/zhan-frontend.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable zhan-api zhan-frontend
sudo systemctl start zhan-api zhan-frontend

echo "Setup done. Check: systemctl status zhan-api zhan-frontend && curl -s https://zhan-ai.kz | head -5"
