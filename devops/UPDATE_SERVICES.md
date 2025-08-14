

The following is a work in progress and is not integreated into our workflow:

## 🔄 Updating a Service When the Network is Already Running

If the Docker Compose network is already running and you need to update a specific service without stopping the entire stack, you can **rebuild and restart the service individually**. This approach ensures minimal downtime and avoids unnecessary restarts of dependent services.

### ⚠️ **Important Note for VS Code Users**
If you are working with **Dev Containers** in VS Code, closing the Dev Container window for one service may **interrupt other local services** unexpectedly. Be mindful of this when stopping or rebuilding individual containers.

### **Updating a Specific Service**
To update a service, use the following command structure:

```sh
docker compose build --no-cache <service_name> && docker compose up -d --no-deps <service_name>
```

- `docker compose build --no-cache <service_name>` → Rebuilds the specified service **without cache**.
- `docker compose up -d --no-deps <service_name>` → Starts (or restarts) only that service **without restarting its dependencies**.

---

### **Examples**
#### **Rebuild and Restart `backend`**
```sh
docker compose build --no-cache backend && docker compose up -d --no-deps backend
```

#### **Rebuild and Restart `backend-scrapers`**
```sh
docker compose build --no-cache backend-scrapers && docker compose up -d --no-deps backend-scrapers
```

#### **Rebuild and Restart `frontend`**
```sh
docker compose build --no-cache frontend && docker compose up -d --no-deps frontend
```

#### **Rebuild and Restart `firestore-emulator`**
```sh
docker compose build --no-cache firestore-emulator && docker compose up -d --no-deps firestore-emulator
```