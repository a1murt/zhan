#!/bin/bash
# Run on server after code update (e.g. pull or rsync). Restarts app services.

set -e
APP_DIR="/var/www/zhan-ai"

cd "$APP_DIR"

# Backend: restart API
sudo systemctl restart zhan-api

# Frontend: rebuild and restart
cd "$APP_DIR/frontend"
npm ci
[[ -f .env.production ]] || echo "NEXT_PUBLIC_API_URL=https://api.zhan-ai.kz/predict/" > .env.production
npm run build
sudo systemctl restart zhan-frontend

echo "Deploy done. zhan-ai.kz and api.zhan-ai.kz"
