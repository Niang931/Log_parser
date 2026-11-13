# The landing page (landing/), built by Vite and served by a small nginx behind Traefik.
#
#   docker build -f deploy/landing.Dockerfile -t logpipe-landing .

FROM node:22-slim AS build

WORKDIR /src
COPY landing/package.json landing/package-lock.json ./
RUN npm ci

COPY landing/ ./
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY deploy/landing.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
