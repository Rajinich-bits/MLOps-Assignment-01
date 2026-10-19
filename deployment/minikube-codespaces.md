# Run the project on Minikube in GitHub Codespaces

Use this guide when the computer you are using has VS Code and a browser but
does not have enough space for Docker or Minikube. Docker, Minikube, the image,
and the Kubernetes cluster run in a remote GitHub Codespace; they are not
installed on the lab computer.

This is still a Minikube setup: Minikube must be available inside the remote
Codespace and uses its Docker daemon. It does not install Minikube on your
local lab machine. Codespaces availability, resource limits, and organization
policies vary. If Docker is unavailable or the Minikube Docker driver is
blocked, ask your instructor for a Kubernetes lab or use the project's local
API/Compose instructions instead.

## 1. Open the repository in a Codespace

1. Push this project to a GitHub repository you can access.
2. Open the repository on GitHub and choose **Code** > **Codespaces** >
   **Create codespace on main** (choose the branch containing the project if
   it is not `main`).
3. Wait for the browser-based VS Code editor to finish starting.
4. Open **Terminal** > **New Terminal**. Run the following commands from the
   repository root for the remaining steps.

Codespaces uses remote compute and storage, so the Minikube download and
container image consume Codespaces resources rather than disk on the lab PC.
Check your account or organization Codespaces quota before starting.

## 2. Check remote Docker access and install the Minikube CLI remotely

The Minikube Docker driver requires a working Docker daemon. Check it first:

```bash
docker info
```

If this fails, stop here: Minikube cannot use the Docker driver until Docker
is available in the Codespace. Do not try to install Docker or Minikube on the
lab PC as a workaround.

The commands below download the Minikube CLI into the Codespace user's home
directory. They do not need `sudo` and do not change the lab computer. This
binary is for an x86-64 (`x86_64`) Linux Codespace:

```bash
uname -m
mkdir -p "$HOME/.local/bin"
curl -fL https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64 \
  -o "$HOME/.local/bin/minikube"
chmod +x "$HOME/.local/bin/minikube"
export PATH="$HOME/.local/bin:$PATH"
minikube version
```

If `uname -m` reports `aarch64` instead, download
`https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-arm64`
in place of the `amd64` URL above. If you open a new terminal later, run
`export PATH="$HOME/.local/bin:$PATH"` there too.

Minikube provides a kubectl wrapper, so a separate kubectl installation is
not needed:

```bash
minikube kubectl -- version --client
```

## 3. Start a small cluster in the Codespace

```bash
minikube start --driver=docker --cpus=2 --memory=3072 --disk-size=12g
minikube status
minikube kubectl -- get nodes
```

The first start downloads the Kubernetes node image and may take a few
minutes. The CPU, memory, and disk settings limit Minikube's requested size;
the Codespace still needs enough available resources to run both Docker and
the cluster.

## 4. Install project dependencies and train the model

The API image expects the trained model at `models/heart_model.joblib`. If
that file is not already present in the Codespace, create it by downloading
the dataset and running training:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/prepare_data.py
python -m heart_ml.train --data data/raw/heart.csv --output models
```

If the repository already contains `models/heart_model.joblib`, training is
optional. Retraining is useful when you want the deployed model to match the
current source and to produce fresh MLflow results.

## 5. Build the API image into Minikube and deploy it

Build the image in Minikube's image store. This avoids relying on Docker Hub
for the project's local image:

```bash
minikube image build -t heart-disease-api:local .
minikube kubectl -- apply -f deployment/k8s.yaml
minikube kubectl -- rollout status deployment/heart-disease-api --timeout=180s
minikube kubectl -- get pods,services
```

The manifest creates one API pod and a `LoadBalancer` service. Its pod checks
the API `/health` endpoint with readiness and liveness probes.

## 6. Open the API from Firefox

Forward the service port from the Codespace. Keep this command running in its
terminal:

```bash
minikube kubectl -- port-forward --address 0.0.0.0 \
  service/heart-disease-api 8000:8000
```

In browser-based VS Code, open the **Ports** tab, find port `8000`, and use
**Open in Browser**. Keep the port private; do not make this educational API
public. In Firefox, open the forwarded URL and visit `/health` or `/docs`.
For example, if Codespaces shows a forwarded URL ending in
`-8000.app.github.dev`, append `/health` or `/docs`.

You can also test from a second Codespace terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected health response:

```json
{"status":"ok"}
```

To test a prediction, use `/docs` and submit a request to `POST /predict`.
This project is an educational demonstration, not a clinical tool.

## 7. Stop the cluster and save space

When finished, stop port forwarding with **Ctrl+C** in its terminal, then run:

```bash
minikube kubectl -- delete -f deployment/k8s.yaml
minikube delete
```

Finally, use the Codespaces menu in GitHub to **Stop codespace**. Stopping
the Codespace releases its running compute; deleting it also removes its
remote files. Save any files you need before deleting it.

## Troubleshooting

- **`docker info` cannot connect:** Docker is not available in this Codespace
  or is restricted by its configuration. Minikube's Docker driver cannot
  start without a daemon.
- **Minikube reports insufficient resources:** Stop other processes in the
  Codespace or choose a larger Codespace if your account allows it. Do not
  increase Minikube's resource values beyond the Codespace's available
  capacity.
- **Pod is not ready / `ImagePullBackOff`:** Check the build completed and
  that the image name is exactly `heart-disease-api:local`:

  ```bash
  minikube kubectl -- describe pod -l app=heart-disease-api
  minikube kubectl -- logs deployment/heart-disease-api
  ```

- **Browser cannot reach the API:** Ensure port-forward is still running,
  port `8000` appears in the Codespaces **Ports** tab, and use the forwarded
  Codespaces URL rather than the Minikube node IP.

## References

- [Minikube Docker driver requirements](https://minikube.sigs.k8s.io/docs/drivers/docker/)
- [GitHub Codespaces quickstart](https://docs.github.com/en/codespaces/quickstart)
- [Forwarding ports in a Codespace](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)
