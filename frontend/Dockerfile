FROM node:18-alpine

WORKDIR /app

# Сначала копируем только конфиги, чтобы Docker закешировал установку библиотек
COPY package*.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем всё остальное
COPY . .

# Пробрасываем порт
EXPOSE 3030

# Команда запуска (она перекроет ту, что в docker-compose, или дополнит её)
CMD ["npm", "start"]
