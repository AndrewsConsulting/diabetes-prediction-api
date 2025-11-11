# Deployment Guide

## Option 1: Railway.app (EASIEST) ⭐

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Choose this repository
5. Done! Auto-deploys in 2 minutes

Your app: `https://your-project-name.railway.app`

## Option 2: Heroku

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

## Option 3: Docker Local

```bash
docker build -t diabetes-api .
docker run -p 8080:8080 diabetes-api
```

## Option 4: Docker Compose

```bash
docker-compose up
```

Visit: `http://localhost:8080`

## Testing After Deploy

```bash
curl https://your-deployed-url/health
curl https://your-deployed-url/status
```
