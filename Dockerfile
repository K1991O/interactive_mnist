# 1. Build stage
FROM node:16 AS build-stage

WORKDIR /app

# Copy the package.json and package-lock.json files for a more efficient cache 
COPY ml-page/package*.json ./

# Install dependencies
RUN npm install

# Copy the source code into the image
COPY ml-page/ ./

# Build the React application
RUN npm run build

# 2. Serve stage
FROM nginx:alpine AS serve-stage

# Copy the built app to the nginx directory
COPY --from=build-stage /app/build /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
