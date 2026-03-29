# Этап 1: Сборка
FROM node:18-alpine as build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# # Этап 2: Раздача статики через Nginx
# FROM nginx:stable-alpha
# COPY --from=build-stage /app/dist /usr/share/nginx/html
# # Если используете React Router, добавьте свой конфиг nginx
# EXPOSE 80
# CMD ["nginx", "-g", "daemon off;"]
