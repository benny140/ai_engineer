# Frontend image — builds the Vite/React app and serves it via nginx.
# Intended for Azure Container Apps eventually; runs on any Docker host locally.
# Build from the frontend/ directory:
#   docker build -f frontend/Dockerfile -t ai-engineer-frontend \
#     --build-arg VITE_API_BASE_URL=https://<backend-ingress-url> frontend

# --- Build stage ---
FROM node:20-alpine AS build

WORKDIR /app

# Backend API URL is baked into the static bundle at build time.
ARG VITE_API_BASE_URL=http://127.0.0.1:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

# --- Serve stage ---
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
