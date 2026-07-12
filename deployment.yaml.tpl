apiVersion: v1
kind: Namespace
metadata:
  name: ${EKS_NAMESPACE}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: demo-xgboost-reg-service
  namespace: ${EKS_NAMESPACE}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: demo-xgboost-reg-service
  template:
    metadata:
      labels:
        app: demo-xgboost-reg-service
    spec:
      containers:
        - name: xgboost-container
          image: ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com/demo-xgboost-reg-service:${IMAGE_TAG}
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: demo-xgboost-reg-service
  namespace: ${EKS_NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app: demo-xgboost-reg-service
  ports:
    - port: 80
      targetPort: 8080
