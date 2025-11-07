BEFORE OPEN SOURCING REMOVE PAT AND ALL SECRETS

## New deploy config

```
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release; echo "$ID")/gpg | \
sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release; echo "$ID") \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
sudo docker pull nginx:latest
docker network create appnet || true
sudo docker run -d \
  --network appnet \
  --name nginx-server \
  -p 80:80 \
  -v /srv/nginx/html:/usr/share/nginx/html \
  -v /srv/nginx/conf:/etc/nginx/conf.d \
  --restart always \
  nginx
sudo systemctl enable docker
sudo docker run -it --rm \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/lib/letsencrypt:/var/lib/letsencrypt \
  -p 80:80 \
  certbot/certbot certonly --standalone \
  -d glo-matcher.brainapi.lumen-labs.ai
sudo tee /srv/nginx/conf/default.conf >/dev/null <<'NGINX'
server {
  listen 80;
  server_name glo-matcher.brainapi.lumen-labs.ai;
  return 301 https://$host$request_uri;
}
server {
  listen 443 ssl;
  server_name glo-matcher.brainapi.lumen-labs.ai;

  ssl_certificate     /etc/letsencrypt/live/glo-matcher.brainapi.lumen-labs.ai/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/glo-matcher.brainapi.lumen-labs.ai/privkey.pem;

  client_max_body_size 50m;

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";

    proxy_connect_timeout   60s;
    proxy_send_timeout      300s;
    proxy_read_timeout      300s;
    send_timeout            300s;
  }
}
NGINX

docker rm -f nginx-server 2>/dev/null || true
docker run -d --name nginx-server \
  -p 80:80 -p 443:443 \
  -v /srv/nginx/html:/usr/share/nginx/html \
  -v /srv/nginx/conf:/etc/nginx/conf.d \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  --restart always \
  nginx:latest

echo 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 | docker login ghcr.io -u ChrisCoder9000 --password-stdin
docker pull ghcr.io/lumen-labs/brainapi:latest
docker run -d \
  --name brainapi \
  -p 8000:8000 \
  --network appnet \
  --restart always \
  ghcr.io/lumen-labs/brainapi:latest

```
