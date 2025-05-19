#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Script Configuration ---
# This script assumes your FastAPI/Uvicorn app runs on 127.0.0.1:8000

echo "--- Nginx and Certbot Setup Script for FastAPI ---"

# --- Step 0: Get Domain Name from Command Line Argument ---
if [ -z "$1" ]; then
    echo "Usage: $0 <your_domain_name>"
    echo "Error: Domain name not provided as a command-line argument."
    exit 1
fi
DOMAIN_NAME="$1"
echo "Domain name set to: $DOMAIN_NAME"

# --- Step 1: Install Nginx, Certbot, and UFW ---
echo "Updating system packages..."
sudo apt update
# sudo apt upgrade -y # Optionally run upgrade if desired, can take time

echo "Installing Nginx, Certbot, Certbot Nginx plugin, and UFW..."
sudo apt install -y \
    nginx \
    certbot \
    python3-certbot-nginx \
    ufw # Added UFW installation

# --- Step 2: Configure Firewall (UFW) ---
echo "Configuring firewall (UFW)..."
# Check if UFW is active. If not, enable it.
if ! command -v ufw &> /dev/null
then
    echo "UFW command could not be found even after attempting installation. Exiting."
    exit 1
fi

if ! sudo ufw status | grep -qw active; then
  echo "UFW is not active. Enabling UFW."
  sudo ufw allow 'OpenSSH' # Ensure SSH access is not blocked BEFORE enabling ufw
  # The following command might prompt for confirmation.
  # To automate, you might use `yes | sudo ufw enable` but that can be risky if not intended.
  # For a script, it's better to inform the user or handle the prompt if possible.
  # For now, we assume the user will handle the (y/n) prompt if it appears.
  sudo ufw enable
else
  echo "UFW is already active."
fi
sudo ufw allow 'Nginx Full' # Allows both HTTP (80) and HTTPS (443)
sudo ufw allow 'OpenSSH'   # Re-ensure SSH is allowed if it was enabled for the first time
sudo ufw status

# --- Step 3 was Get Domain Name, now handled in Step 0 ---

# --- Step 4: Configure Nginx as a Reverse Proxy ---
echo "Configuring Nginx for domain: $DOMAIN_NAME..."

# Define the Nginx server block configuration content
NGINX_CONFIG_CONTENT="server {
    listen 80;
    listen [::]:80;

    server_name $DOMAIN_NAME; # Add www.$DOMAIN_NAME here if you want Nginx to handle it initially

    location / {
        proxy_pass http://127.0.0.1:8000; # Assumes Uvicorn runs on localhost port 8000
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support (uncomment if your FastAPI app uses WebSockets)
        # proxy_http_version 1.1;
        # proxy_set_header Upgrade \$http_upgrade;
        # proxy_set_header Connection \"upgrade\";
    }

    # Optional: If you want to serve static files directly via Nginx from your FastAPI app's static dir
    # location /static/ {
    #     alias /path/to/your/fastapi_app_project/static/; # Adjust this path accordingly
    # }

    # Certbot may add ACME challenge location blocks here later
}"

# Create Nginx server block configuration file
echo "$NGINX_CONFIG_CONTENT" | sudo tee /etc/nginx/sites-available/"$DOMAIN_NAME" > /dev/null

# Disable the default Nginx site if it exists and is enabled
if [ -L /etc/nginx/sites-enabled/default ]; then
    echo "Disabling default Nginx site..."
    sudo unlink /etc/nginx/sites-enabled/default
fi

# Enable the new server block
if [ ! -L /etc/nginx/sites-enabled/"$DOMAIN_NAME" ]; then
    echo "Enabling Nginx site for $DOMAIN_NAME..."
    sudo ln -s /etc/nginx/sites-available/"$DOMAIN_NAME" /etc/nginx/sites-enabled/
else
    echo "Nginx site for $DOMAIN_NAME already seems enabled."
fi

echo "Testing Nginx configuration..."
if sudo nginx -t; then
    echo "Nginx configuration is OK."
    echo "Restarting Nginx..."
    sudo systemctl restart nginx
else
    echo "Nginx configuration test failed. Please check /etc/nginx/sites-available/$DOMAIN_NAME and Nginx logs."
    exit 1
fi

# --- Step 5: Obtain SSL Certificate with Certbot ---
echo "Obtaining SSL certificate for $DOMAIN_NAME using Certbot..."
echo "Certbot will ask for your email and to agree to terms."
echo "It will also ask if you want to redirect HTTP to HTTPS (recommended)."

# Run Certbot
# The --nginx plugin will attempt to automatically modify the Nginx config for SSL.
# The -d flag specifies the domain(s). If you want www, add -d www.yourdomain.com
# Using --redirect is generally recommended.
# Using --non-interactive --agree-tos -m your_email@example.com for full automation is possible but requires email.
# For now, keeping it interactive for email and TOS.
sudo certbot --nginx -d "$DOMAIN_NAME" # Add additional -d flags for www or other SANs if needed

echo "Certbot process finished."
echo "Verifying Nginx status after Certbot..."
sudo systemctl status nginx --no-pager # Check if Nginx is still running fine

echo "Testing Certbot auto-renewal..."
sudo certbot renew --dry-run

echo "--- Setup Complete ---"
echo "Your FastAPI application should now be accessible via https://$DOMAIN_NAME"
echo "Make sure your FastAPI application (e.g., Uvicorn) is running on 127.0.0.1:8000."
echo "Consider setting up your FastAPI application to run as a systemd service for production."
echo "Example Gunicorn command for production (run from your FastAPI project venv):"
echo "gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app -b 127.0.0.1:8000"

