# Деплой ZhanAI на 89.218.178.215 (zhan-ai.kz)

Сервер: **89.218.178.215**  
Пользователь: **administrator**  
Домен: **zhan-ai.kz** (фронт), **api.zhan-ai.kz** (API). SSL — Let's Encrypt.

---

## 1. DNS

В панели домена **zhan-ai.kz** добавьте A-записи на IP сервера:

| Тип | Имя | Значение    | TTL |
|-----|-----|-------------|-----|
| A   | @   | 89.218.178.215 | 300 |
| A   | www | 89.218.178.215 | 300 |
| A   | api | 89.218.178.215 | 300 |

Проверка (через несколько минут):

```bash
nslookup zhan-ai.kz
nslookup api.zhan-ai.kz
```

---

## 2. Подключение к серверу

```bash
ssh administrator@89.218.178.215
```

(Используйте ваш пароль или ключ.)

---

## 3. Загрузка проекта на сервер

**Вариант A — через rsync с вашего ПК (из папки `zhan`):**

```bash
# С Windows (PowerShell) — через WSL или установленный rsync/ssh
rsync -avz --exclude "node_modules" --exclude "frontend/node_modules" --exclude ".git" --exclude "data" --exclude "__pycache__" ./ administrator@89.218.178.215:/var/www/zhan-ai/
```

Создать каталог на сервере один раз:

```bash
ssh administrator@89.218.178.215 "sudo mkdir -p /var/www/zhan-ai && sudo chown administrator:administrator /var/www/zhan-ai"
```

**Вариант B — клонирование репозитория на сервере** (если проект в Git):

```bash
ssh administrator@89.218.178.215
sudo mkdir -p /var/www/zhan-ai && sudo chown administrator:administrator /var/www/zhan-ai
git clone <URL вашего репо> /var/www/zhan-ai
```

**Обязательно скопировать чекпоинт модели:**

- Файл `checkpoints/odir_best.pt` должен лежать на сервере в `/var/www/zhan-ai/checkpoints/odir_best.pt` (скопируйте вручную или добавьте в rsync).

---

## 4. Первичная установка на сервере

На сервере выполните:

```bash
ssh administrator@89.218.178.215
cd /var/www/zhan-ai
chmod +x deploy/setup-server.sh
./deploy/setup-server.sh
```

В скрипте при запросе SSL (certbot) укажите email: `admin@zhan-ai.kz` (или замените в скрипте `-m admin@zhan-ai.kz` на свой).

Если репозиторий не клонировали, а залили через rsync — перед запуском убедитесь, что в `/var/www/zhan-ai` есть папки `frontend`, `src`, `deploy`, `requirements.txt` и файл `checkpoints/odir_best.pt`.

---

## 5. SSL (если не сработало в setup-server.sh)

Когда DNS уже указывает на 89.218.178.215:

```bash
sudo certbot --nginx -d zhan-ai.kz -d www.zhan-ai.kz -d api.zhan-ai.kz
```

Введите email, согласитесь с условиями. Certbot сам настроит nginx на HTTPS.

---

## 6. Обновление после изменений кода (деплой)

На сервере:

```bash
cd /var/www/zhan-ai
# если через git: git pull
# если через rsync: снова выполните rsync с ПК
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

Или вручную:

```bash
cd /var/www/zhan-ai
./venv/bin/pip install -r requirements.txt
sudo systemctl restart zhan-api

cd frontend
npm ci
echo "NEXT_PUBLIC_API_URL=https://api.zhan-ai.kz/predict/" > .env.production
npm run build
sudo systemctl restart zhan-frontend
```

---

## 7. Проверка

- Сайт: https://zhan-ai.kz  
- API: https://api.zhan-ai.kz/health  
- Документация API: https://api.zhan-ai.kz/docs  

Логи:

```bash
sudo journalctl -u zhan-api -f
sudo journalctl -u zhan-frontend -f
sudo tail -f /var/log/nginx/error.log
```

---

## Краткий чеклист

1. DNS: A-записи для `zhan-ai.kz`, `www`, `api` → 89.218.178.215  
2. Скопировать проект в `/var/www/zhan-ai` (в т.ч. `checkpoints/odir_best.pt`)  
3. Запустить `deploy/setup-server.sh` на сервере  
4. При необходимости запустить certbot для SSL  
5. Открыть https://zhan-ai.kz и https://api.zhan-ai.kz/health  
