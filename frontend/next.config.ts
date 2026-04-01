import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  experimental: {
    // Required for larger file uploads when proxying /api/* to FastAPI.
    // Default is 10MB, which fails on PDFs like CASA.pdf (17.54MB).
    proxyClientMaxBodySize: 50 * 1024 * 1024,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
