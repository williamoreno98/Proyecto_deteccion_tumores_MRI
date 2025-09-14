# ---- Build (Vite) ----
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# ---- Runtime (Nginx + proxy) ----
FROM nginx:alpine
# envsubst para inyectar variables en nginx.conf
RUN apk add --no-cache gettext
# copiamos el sitio estático
COPY --from=build /app/dist /usr/share/nginx/html
# plantilla y script de arranque
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 8080
CMD ["/start.sh"]
