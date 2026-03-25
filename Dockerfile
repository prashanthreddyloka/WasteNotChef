FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
COPY client/package*.json ./client/
RUN npm install && npm install --prefix client

COPY . .

RUN npm run build

EXPOSE 4000 5173

CMD ["npm", "run", "dev"]
