# Frontend: Next.js, built with the API URL baked in at build time.
#
# IMPORTANT: NEXT_PUBLIC_* variables are inlined into the JS bundle at build
# time, not read at container start. If you change NEXT_PUBLIC_API_URL you
# must rebuild this image, not just restart the container. This is also why
# the value must be a URL the *browser* can reach — pages call the API
# client-side (see lib/api.js) — not an internal Docker service name like
# "http://backend:8000", which only resolves inside the Docker network.

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN addgroup --system nodejs && adduser --system --ingroup nodejs nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
