#!/usr/bin/env sh
set -eu
# Defaults
: "${PORT:=8080}"
: "${API_URL:?Set API_URL env var in Railway (e.g. https://proyectodetecciontumoresmri-production.up.railway.app)}"
# Renderiza la plantilla con variables
envsubst '${API_URL} ${PORT}' \
  < /etc/nginx/templates/nginx.conf.template \
  > /etc/nginx/conf.d/default.conf
# Arranca Nginx
nginx -g 'daemon off;'
