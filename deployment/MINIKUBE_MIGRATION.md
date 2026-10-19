# Moving the project to Minikube

This guide runs the **API, Prometheus, and Grafana together in Minikube**. It
uses the QEMU VM driver with Minikube's `containerd` image store. Docker Desktop
and Colima are not used for this setup. Model training and MLflow remain local
on the Mac; the trained model is copied into the API container image.

## What changed

- `deployment/minikube-stack.yaml` defines the API, Prometheus, and Grafana
  Deployments and internal `ClusterIP` Services in the `mlops` namespace.
- `kustomization.yaml` generates ConfigMaps from the existing Grafana dashboard
  and provisioning files, plus the Minikube-specific Prometheus scrape config.
  `kubectl apply -k .` refreshes these ConfigMaps when the files change.
- `monitoring/prometheus-minikube.yml` scrapes the API at
  `heart-disease-api:8000`, which is Kubernetes service DNS. Compose keeps its
  separate `api:8000` configuration.
- The Grafana password is supplied in a Kubernetes Secret, rather than stored
  in the deployment manifest.
- The Minikube stack uses ephemeral storage for Grafana and Prometheus. The
  dashboard and datasource are provisioned from files, but Prometheus history
  and Grafana state are reset if those pods are recreated.
- `deployment/k8s.yaml` and `docker-compose.yml` remain available for the
  previous Colima/Minikube-API-only and Compose workflows.
- This workspace already had an API-only deployment in the Minikube `default`
  namespace. It was left untouched; the new complete stack runs separately in
  `mlops`.

## 1. Check prerequisites

On Apple Silicon macOS, install the command-line tools with Homebrew if needed:

```bash
brew install minikube qemu kubectl
```

Check that QEMU and Minikube are on your PATH:

```bash
qemu-system-aarch64 --version
minikube version
kubectl version --client
```

The tested local Minikube profile is the QEMU driver with the `containerd`
runtime. Start the existing profile with:

```bash
minikube start
```

For a fresh cluster, create it explicitly with modest resources:

```bash
minikube start --driver=qemu --network=builtin --container-runtime=containerd \
  --cpus=2 --memory=4096 --disk-size=20g
```

Verify the cluster before deploying:

```bash
minikube status
kubectl config current-context
kubectl get nodes
```

The current kubectl context should be `minikube`, and the node should be
`Ready`. The `builtin` QEMU network works with `kubectl port-forward`; this
guide does not rely on `minikube tunnel`.

## 2. Prepare and train the model

Run these commands from the repository root. Activate the existing virtual
environment; create/install it only if it does not exist yet.

```bash
cd /Users/rci968/MLOps-Assignment-01
source .venv/bin/activate
```

If `data/raw/heart.csv` is absent, download it:

```bash
python scripts/download_data.py
```

Prepare the cleaned data, create EDA figures, run tests, and train both models:

```bash
python scripts/prepare_data.py
python scripts/eda.py
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
python -m heart_ml.train --data data/raw/heart.csv --output models
```

The training command prints the selected model and metrics, writes
`models/heart_model.joblib` and `models/heart_model.pkl`, and records the
comparison and selected-model artifacts in the local MLflow store.

To inspect the model selection summary:

```bash
python -c 'import json; d=json.load(open("reports/metrics.json")); print("Selected:", d["selected_model"]); print("CV ROC-AUC:", d["selected_cv_roc_auc_mean"]); [print(m["model"], "accuracy=", round(m["accuracy"], 4), "precision=", round(m["precision"], 4), "recall=", round(m["recall"], 4), "ROC-AUC=", round(m["roc_auc"], 4)) for m in d["models"]]'
```

Optionally run the MLflow UI locally in another terminal:

```bash
cd /Users/rci968/MLOps-Assignment-01
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and inspect the newest `selected_model` run. MLflow
is not deployed into Minikube by these changes.

## 3. Build the API image in Minikube

The Dockerfile copies the trained joblib model into the image. Build directly
inside Minikube so Kubernetes can use the image without a registry or a Docker
daemon on macOS:

```bash
minikube image build -t heart-disease-api:local .
minikube image ls | grep heart-disease-api
```

If the Minikube VM cannot download the base image, check network access and
retry the image build after connectivity is restored.

## 4. Create the namespace and Grafana credential

Create the namespace:

```bash
kubectl apply -f deployment/minikube-namespace.yaml
```

Create a local Grafana password and store it in a Kubernetes Secret. Replace
the value below with a password of your choice; it is only for this local
demonstration:

```bash
kubectl create secret generic grafana-admin \
  --namespace mlops \
  --from-literal=admin-user=admin \
  --from-literal=admin-password='choose-a-local-password' \
  --dry-run=client -o yaml | kubectl apply -f -
```

Do not expose this demo Grafana service to the public internet.

## 5. Deploy the stack

From the repository root, apply the Kubernetes resources and generated
ConfigMaps:

```bash
kubectl apply -k .
kubectl rollout status deployment/heart-disease-api -n mlops --timeout=180s
kubectl rollout status deployment/prometheus -n mlops --timeout=180s
kubectl rollout status deployment/grafana -n mlops --timeout=180s
kubectl get pods,services -n mlops -o wide
```

All three pods should show `Running` and `1/1` ready. The Services are
cluster-internal; local access is provided with port-forwarding below.

## 6. Forward local ports

Open three terminals from the project directory (or any directory) and keep
each command running in its own terminal:

```bash
kubectl port-forward --address 127.0.0.1 -n mlops service/heart-disease-api 8000:8000
```

```bash
kubectl port-forward --address 127.0.0.1 -n mlops service/prometheus 9090:9090
```

```bash
kubectl port-forward --address 127.0.0.1 -n mlops service/grafana 3000:3000
```

## 7. Verify the API, metrics, and dashboard

In a fourth terminal, check the API health:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected: HTTP `200` with `{"status":"ok"}`. Submit a sample prediction:

```bash
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
```

The response contains `prediction`, `confidence`, and `positive_probability`.
The exact prediction depends on the trained model; compare the output with
your actual run rather than treating it as a fixed expected clinical result.

Check that the API's Prometheus target is being scraped:

```bash
curl -sS 'http://127.0.0.1:9090/api/v1/query?query=up%7Bjob%3D%22heart-disease-api%22%7D'
```

The query result should have value `"1"`. The Prometheus target page is
<http://127.0.0.1:9090/targets>. After making a few API requests, the raw
metrics endpoint is <http://127.0.0.1:8000/metrics>.

Open Grafana at <http://127.0.0.1:3000>, sign in as `admin` with the password
created in step 4, then open the **Heart Disease API** dashboard in the
**MLOps Demo** folder. Generate requests and refresh the dashboard to populate
request-rate and latency panels:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
```

## 8. Troubleshooting and cleanup

- Check pod logs with `kubectl logs -n mlops deployment/heart-disease-api`,
  `kubectl logs -n mlops deployment/prometheus`, or
  `kubectl logs -n mlops deployment/grafana`.
- If image changes do not appear, rebuild with `minikube image build -t
  heart-disease-api:local .` and restart the API deployment with
  `kubectl rollout restart deployment/heart-disease-api -n mlops`.
- If a local port is already used, stop its other service or choose a different
  host port in the `kubectl port-forward` command (for example `8001:8000`).
- Stop each port-forward with `Ctrl+C`.
- Remove only this stack's resources with:

  ```bash
  kubectl delete -k .
  kubectl delete namespace mlops
  ```

- Keep the Minikube VM installed but stop it when not in use:

  ```bash
  minikube stop
  ```

## What did not move into Minikube

The local MLflow tracking server and SQLite database remain on the Mac, and
the training process still runs in `.venv`. This keeps cluster resource use
lower and preserves the existing experiment artifacts. The API image contains
the selected trained model. Docker Compose remains an optional alternative for
running the API and monitoring stack when Docker Desktop or Colima is available.

## Verification performed in this workspace

On 2026-10-19, the stack was built and deployed on Minikube 1.39.0 with the
QEMU2 driver, Kubernetes 1.37.0, and containerd. No Docker Desktop or Colima
engine was used. The API, Prometheus, and Grafana pods all reached `1/1`
readiness.

The Prometheus target `heart-disease-api:8000` returned `up=1`; API request-rate
and p95-latency dashboard panels displayed data in Grafana. Grafana's
provisioned Prometheus datasource passed its health check and the
`Heart Disease API` dashboard appeared in search.

The verified training run selected Logistic Regression by mean five-fold
cross-validation ROC-AUC:

| Model | Mean CV ROC-AUC (SD) | Holdout accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression (selected) | 0.9008 (0.0145) | 0.8852 | 0.8387 | 0.9286 | 0.9665 |
| Random Forest | 0.8914 (0.0257) | 0.8689 | 0.8333 | 0.8929 | 0.9578 |

For the documented sample record, the Minikube API returned prediction `0`,
confidence `0.9217`, and positive probability `0.0783`; this matched inference
from the locally saved `models/heart_model.joblib`. Ruff checks and all 13
pytest tests passed, and `pip check` found no broken requirements. These are
educational demo results, not clinical validation.
