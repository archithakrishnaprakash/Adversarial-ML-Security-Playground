/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // "standalone" traces only the files actually needed at runtime into
  // .next/standalone, so the production Docker image doesn't have to ship
  // the full node_modules tree — smaller image, faster deploys.
  output: "standalone",
};

module.exports = nextConfig;
