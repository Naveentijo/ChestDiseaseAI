# Base Node.js image
FROM node:20-alpine

WORKDIR /app

# Copy dependency configs
COPY package*.json ./

# Install npm packages
RUN npm ci

# Copy all source files
COPY . .

# Set production environment variable for build context
ENV NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production

# Compile Next.js and Tailwind CSS production bundles
RUN npm run build

# Expose port
EXPOSE 3000

# Start Next.js server
CMD ["npm", "start"]
