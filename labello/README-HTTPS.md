# Camera App - HTTPS Setup

## Setup for New Developers

If you're cloning this repository, you'll need to generate SSL certificates first:

```bash
# Clone and install dependencies
git clone <repository-url>
cd camera-app
npm install

# Generate SSL certificates for HTTPS
mkdir -p .ssl
cd .ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
  -subj "/C=US/ST=State/L=City/O=Dev/OU=Dev/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | awk '{print $2}' | head -1)"
cd ..

# Now you can run the HTTPS server
npm run dev:https
```

## Problem Solved

The camera access issue on mobile devices has been resolved by setting up HTTPS for local development. The error "undefined is not an object (evaluating 'navigator.mediaDevices.getUserMedia')" occurred because modern browsers require HTTPS to access camera and microphone for security reasons.

## Solution

1. **Self-signed SSL certificates** have been generated in the `.ssl/` directory
2. **Custom HTTPS server** (`server.mjs`) has been created for development
3. **Enhanced error handling** provides better user feedback
4. **Network access** is configured to work on any local network

## Usage

### For Desktop Development
```bash
npm run dev        # HTTP (localhost only)
npm run dev:https  # HTTPS (works on mobile)
```

### For Mobile Testing
1. Start the HTTPS server: `npm run dev:https`
2. **Find your computer's IP address:**
   - **macOS/Linux**: `ifconfig | grep 'inet ' | grep -v 127.0.0.1`
   - **Windows**: `ipconfig` (look for IPv4 Address)
   - **Or check the terminal output** when starting the server - it shows the Network URL
3. On your mobile device, navigate to: `https://YOUR_IP_ADDRESS:3000`
4. **Important**: Your browser will show a security warning because it's a self-signed certificate
5. Click "Advanced" → "Proceed to [IP] (unsafe)" or similar option
6. The camera should now work properly on mobile!

## Available URLs

- **Desktop (localhost)**: https://localhost:3000
- **Mobile/Network**: https://YOUR_IP_ADDRESS:3000
- **HTTP (desktop only)**: http://localhost:3000

> 💡 **Tip**: The server automatically displays the correct network URL when you run `npm run dev:https`

## Security Warning

When accessing the HTTPS version, you'll see a browser warning about the self-signed certificate. This is normal for development. Click through the warning to proceed.

## Features Fixed

✅ Camera access on mobile devices  
✅ Better error messages for different failure scenarios  
✅ Network access from local devices  
✅ HTTPS requirement for getUserMedia API

## Error Messages

The app now provides specific error messages:
- "Navigation non supportée. Utilisez HTTPS pour accéder à la caméra."
- "Permission caméra refusée. Veuillez autoriser l'accès."
- "Aucune caméra trouvée sur cet appareil."
- "Caméra déjà utilisée par une autre application."

## Important: IP Address Changes

⚠️ **Your IP address will change when:**
- You connect to a different WiFi network
- Your router restarts and assigns new IPs
- Your colleagues use this on their own networks
- Your computer gets a new DHCP lease

**Solution**: Always check the terminal output when running `npm run dev:https` - it shows the current network URL to use.

## Development Notes

- The `.ssl/` directory contains the generated certificates
- `server.mjs` is the custom HTTPS development server
- Camera permissions must be granted when prompted
- Use the HTTPS version for mobile testing
- Check terminal output for the correct IP address each time