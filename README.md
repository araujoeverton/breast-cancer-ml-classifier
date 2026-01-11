<img align="right" src="https://raw.githubusercontent.com/araujoeverton/breast-cancer-ml-classifier/refs/heads/main/assets/project-cover.jpg" width="1080"/><br>

<div align="center">

<br>
...

### Production-Ready MLOps Solution for Breast Cancer Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/Model-ResNet--50-orange.svg)](https://arxiv.org/abs/1512.03385)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-yellow.svg)](https://aws.amazon.com/sagemaker/)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-purple.svg)](https://www.terraform.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-overview) • [Features](#-key-features) • [Architecture](#-architecture) • [Setup](#-setup--installation) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## Overview

This project delivers an **end-to-end MLOps solution** for breast cancer classification using deep learning and AWS cloud services. Built with production best practices, it combines cutting-edge machine learning with enterprise-grade infrastructure automation.

### Business Value

- **Early Detection Support**: Assists healthcare professionals in identifying potentially malignant breast lesions from medical imaging
- **Scalable Infrastructure**: Event-driven serverless architecture that scales automatically with demand
- **Cost Optimization**: Pay-per-use model with automated resource management reduces operational costs
- **Rapid Deployment**: Infrastructure-as-Code enables consistent deployments across environments in minutes
- **Production Ready**: Implements MLOps best practices including automated training, evaluation, and monitoring

### Use Cases

✅ Clinical decision support for radiologists
✅ Second-opinion analysis for breast imaging
✅ Research and development in medical AI
✅ Educational tool for ML engineering in healthcare
✅ Prototype for hospital digital transformation initiatives

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Deep Learning** | ResNet-50 architecture with transfer learning for accurate classification |
| **Cloud-Native** | Fully deployed on AWS with SageMaker, Lambda, S3, and EventBridge |
| **Serverless Inference** | Auto-scaling endpoint with 4GB memory and configurable concurrency |
| **Event-Driven** | Automated inference triggered by S3 uploads via EventBridge |
| **Infrastructure-as-Code** | Complete Terraform modules for reproducible deployments |
| **Hyperparameter Tuning** | Automated optimization using Bayesian search strategies |
| **Model Evaluation** | Comprehensive metrics including precision, recall, F1, and confusion matrix |
| **Cost-Effective** | GPU resources used only during training, serverless for inference |

---

## Architecture

### System Overview

This solution implements a modern MLOps architecture with clear separation between training and inference workflows:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Training Pipeline                      │
│  Kaggle Dataset → S3 → SageMaker Training → Hyperparameter      │
│  Tuning → Model Evaluation → Model Registry → Serverless Deploy │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Inference Pipeline (Event-Driven)            │
│  Image Upload (S3) → EventBridge Trigger → Lambda Function →    │
│  SageMaker Endpoint → Diagnosis Result (BENIGN/MALIGNANT)       │
└─────────────────────────────────────────────────────────────────┘
```

> **📊 Architecture Diagram**: _[Coming Soon] - Full system architecture showing AWS service integration_

### Technology Stack

**Machine Learning**
- Framework: AWS SageMaker Built-in Image Classification (MXNet-based)
- Model: ResNet-50 (Residual Neural Network with 50 layers)
- Training: GPU-accelerated instances (ml.g4dn.xlarge)

**Cloud Infrastructure (AWS)**
- **SageMaker**: Model training, hyperparameter tuning, serverless hosting
- **S3**: Data lake for images, manifests, and model artifacts
- **Lambda**: Serverless inference orchestration (Python 3.9)
- **EventBridge**: Event-driven automation for inference triggers
- **IAM**: Role-based access control and least-privilege policies
- **CloudWatch**: Logging, monitoring, and observability

**DevOps & IaC**
- **Terraform**: Infrastructure provisioning and management (~5.0)
- **Jupyter Notebooks**: Interactive ML experimentation and documentation
- **Git**: Version control with comprehensive .gitignore

---

## ML Pipeline Workflow

### 1️⃣ Data Preprocessing
- **Dataset**: CBIS-DDSM Breast Cancer Image Dataset (Kaggle)
- **Processing**: Path normalization, label mapping, stratified splitting
- **Output**: SageMaker-compatible manifest files (.lst format)
- **Storage**: Organized upload to S3 with prefix-based structure

### 2️⃣ Model Training
- **Algorithm**: ResNet-50 with transfer learning from ImageNet
- **Configuration**: 224x224 RGB images, 20 epochs, Adam optimizer
- **Features**: Early stopping, validation accuracy tracking
- **Instance**: ml.g4dn.xlarge (GPU-accelerated, ~$0.736/hour)

### 3️⃣ Hyperparameter Tuning
- **Strategy**: Random Search with 5 jobs (max 2 parallel)
- **Parameters**: Learning rate, batch size, optimizer, momentum
- **Objective**: Maximize validation accuracy
- **Early Stopping**: Enabled (patience=3, min_epochs=5)

### 4️⃣ Model Evaluation
- **Metrics**: Precision, Recall, F1-Score, Confusion Matrix
- **Process**: Automated best-model selection, test endpoint deployment
- **Validation**: Full classification report on validation set
- **Cleanup**: Automatic endpoint deletion to avoid costs

### 5️⃣ Production Deployment
- **Type**: SageMaker Serverless Inference Endpoint
- **Configuration**: 4GB memory, max concurrency of 5
- **Integration**: Lambda function for event-driven inference
- **Output**: Binary classification with confidence scores

> **🔄 Pipeline Diagram**: _[Coming Soon] - Visual workflow from data ingestion to deployment_

---

## Model Information

### Dataset: CBIS-DDSM
- **Source**: Curated Breast Imaging Subset of DDSM
- **Type**: Mammography images (DICOM converted to JPG)
- **Classes**: Binary classification
  - Class 0: BENIGN (includes BENIGN_WITHOUT_CALLBACK)
  - Class 1: MALIGNANT
- **Split**: 80% training, 20% validation (stratified)

### Model Architecture
- **Base Model**: ResNet-50 (pre-trained on ImageNet)
- **Approach**: Transfer learning with fine-tuning
- **Input Shape**: 3 × 224 × 224 (RGB channels)
- **Output**: Binary classification logits
- **Decision Threshold**: Probability > 0.5 → MALIGNANT

### Performance Metrics
> **📈 Performance Chart**: _[Coming Soon] - Model accuracy, precision, recall across training epochs_

---

## Prerequisites

Before starting, ensure you have:

### Required Accounts & Access
- ☑️ **AWS Account** with appropriate permissions
  - SageMaker full access
  - S3 read/write permissions
  - Lambda and EventBridge access
  - IAM role creation capabilities
- ☑️ **Kaggle Account** for dataset access
  - API credentials (kaggle.json)

### Required Tools
- ☑️ **Python 3.9+** installed
- ☑️ **AWS CLI** configured with credentials
- ☑️ **Terraform ~5.0** installed
- ☑️ **Git** for repository cloning
- ☑️ **Jupyter Notebook** environment (Jupyter Lab or VS Code)

### IAM Permissions Needed
```
- sagemaker:*
- s3:*
- lambda:*
- events:*
- iam:CreateRole
- iam:AttachRolePolicy
- ssm:PutParameter
- logs:CreateLogGroup
```

---

## Setup & Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/araujoeverton/breast-cancer-ml-classifier.git
cd breast-cancer-ml-classifier
```

### Step 2: Configure AWS Credentials
```bash
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Set default region: us-east-1
# Set default output format: json
```

Verify configuration:
```bash
aws sts get-caller-identity
```

### Step 3: Setup Kaggle API
1. Create Kaggle API credentials:
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Click "Create New API Token"
   - Save `kaggle.json` to your system

2. Configure Kaggle credentials:
   ```bash
   # Linux/Mac
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json

   # Windows
   # Place kaggle.json in: C:\Users\<YourUsername>\.kaggle\
   ```

### Step 4: Create Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 5: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies installed:
- `boto3` - AWS SDK
- `sagemaker` - SageMaker Python SDK
- `h5py`, `numpy` - Data processing
- `tqdm` - Progress tracking
- `matplotlib` - Visualization
- `opencv-python` - Image processing
- `pydicom` - Medical imaging

### Step 6: Configure Terraform Variables
1. Navigate to Terraform directory:
   ```bash
   cd infra/environment/dev
   ```

2. Edit [terraform.tfvars.tf](infra/environment/dev/terraform.tfvars.tf):
   ```hcl
   aws_region     = "us-east-1"
   aws_account_id = "YOUR_AWS_ACCOUNT_ID"  # Replace with your account ID
   project_name   = "cbis-ddsm"
   environment    = "dev"
   endpoint_name  = "cbis-ddsm-serverless-endpoint"
   ```

3. Get your AWS Account ID:
   ```bash
   aws sts get-caller-identity --query Account --output text
   ```

### Step 7: Deploy Infrastructure with Terraform
```bash
# Navigate to infrastructure directory
cd infra

# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Deploy infrastructure
terraform apply

# Type 'yes' when prompted to confirm
```

**Resources Created:**
- S3 bucket for data and models
- IAM roles for SageMaker and Lambda
- Lambda function for inference
- EventBridge rule for S3 triggers
- SSM parameters for configuration

### Step 8: Run ML Notebooks
Navigate to [app/src/models/](app/src/models/) and execute notebooks in order:

```bash
jupyter lab app/src/models/
```

**Execution Order:**
1. **[01_preprocessing.ipynb](app/src/models/01_preprocessing.ipynb)** - Download and preprocess CBIS-DDSM dataset
2. **[02_resnet50_train_model.ipynb](app/src/models/02_resnet50_train_model.ipynb)** - Train ResNet-50 with hyperparameter tuning
3. **[03_evaluate_model.ipynb](app/src/models/03_evaluate_model.ipynb)** - Evaluate model performance
4. **[04_deploy_endpoint.ipynb](app/src/models/04_deploy_endpoint.ipynb)** - Deploy serverless inference endpoint
5. **[05_monitoring_drift.ipynb](app/src/models/05_monitoring_drift.ipynb)** - _(Optional)_ Model drift monitoring
6. **[06_mlops_pipeline.ipynb](app/src/models/06_mlops_pipeline.ipynb)** - _(Optional)_ Complete MLOps automation

---

## 📁 Project Structure

```
breast-cancer-ml-classifier/
├── 📂 app/                              # Application code
│   └── src/
│       ├── data_utils/                  # Data processing utilities
│       │   ├── commons.py               # Download, extract, preprocessing
│       │   └── __init__.py
│       ├── models/                      # ML pipeline notebooks
│       │   ├── 01_preprocessing.ipynb           # Data preparation
│       │   ├── 02_resnet50_train_model.ipynb    # Model training
│       │   ├── 03_evaluate_model.ipynb          # Model evaluation
│       │   ├── 04_deploy_endpoint.ipynb         # Deployment
│       │   ├── 05_monitoring_drift.ipynb        # Drift detection
│       │   └── 06_mlops_pipeline.ipynb          # Pipeline automation
│       └── lambda/
│           └── lambda_function_inference.py     # Serverless inference
│
├── 📂 infra/                            # Infrastructure-as-Code (Terraform)
│   ├── main.tf                          # Main orchestration
│   ├── provider.tf                      # AWS provider config
│   ├── variables.tf                     # Input variables
│   ├── parameters.tf                    # SSM parameter store
│   ├── outputs.tf                       # Export values
│   ├── modules/
│   │   ├── s3/                          # S3 bucket module
│   │   ├── iam/                         # IAM roles & policies
│   │   ├── lambda/                      # Lambda function packaging
│   │   ├── sagemaker/                   # SageMaker endpoint config
│   │   └── eventbridge/                 # Event-driven automation
│   └── environment/
│       └── dev/
│           └── terraform.tfvars.tf      # Dev environment config
│
├── 📂 assets/                           # Project images and diagrams
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
└── 📄 LICENSE                           # Project license
```

---

## Usage

### Training a New Model

1. Ensure infrastructure is deployed (see Setup Step 7)

2. Run preprocessing:
   ```python
   # Execute 01_preprocessing.ipynb
   # This will download CBIS-DDSM from Kaggle and upload to S3
   ```

3. Start training with hyperparameter tuning:
   ```python
   # Execute 02_resnet50_train_model.ipynb
   # Training time: ~30-60 minutes on ml.g4dn.xlarge
   ```

4. Evaluate model performance:
   ```python
   # Execute 03_evaluate_model.ipynb
   # Generates classification report and confusion matrix
   ```

### Deploying the Endpoint

```python
# Execute 04_deploy_endpoint.ipynb

# The endpoint will be created as:
# - Name: cbis-ddsm-serverless-endpoint
# - Type: Serverless
# - Memory: 4GB
# - Max Concurrency: 5
```

### Running Inference

**Option 1: Direct SageMaker Endpoint**
```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Prepare image (base64 encoded or S3 path)
response = runtime.invoke_endpoint(
    EndpointName='cbis-ddsm-serverless-endpoint',
    ContentType='application/x-image',
    Body=image_bytes
)

result = json.loads(response['Body'].read())
probability = result[0]  # Cancer probability (0.0-1.0)

diagnosis = "MALIGNANT" if probability > 0.5 else "BENIGN"
confidence = max(probability, 1 - probability)
```

**Option 2: Event-Driven (Automated)**
```bash
# Upload image to S3 entrada/ prefix
aws s3 cp mammogram.jpg s3://cbis-ddsm-dev-data-{account_id}/entrada/

# EventBridge automatically triggers Lambda
# Lambda invokes SageMaker endpoint
# Result logged to CloudWatch
```

### Monitoring Results

```bash
# View Lambda logs
aws logs tail /aws/lambda/cbis-ddsm-inference --follow

# Check endpoint metrics
aws cloudwatch get-metric-statistics \
    --namespace AWS/SageMaker \
    --metric-name ModelLatency \
    --dimensions Name=EndpointName,Value=cbis-ddsm-serverless-endpoint \
    --start-time 2026-01-11T00:00:00Z \
    --end-time 2026-01-11T23:59:59Z \
    --period 3600 \
    --statistics Average
```

---

## Cost Considerations

### Estimated AWS Costs (Monthly)

| Resource | Usage | Estimated Cost |
|----------|-------|----------------|
| **SageMaker Training** | ~1 hour/month (ml.g4dn.xlarge) | $0.74 |
| **SageMaker Serverless** | 1000 inferences, 10s avg | $5-10 |
| **S3 Storage** | 50GB dataset + models | $1.15 |
| **Lambda** | 1000 invocations | $0.20 |
| **EventBridge** | 1000 events | $0.00 (free tier) |
| **CloudWatch Logs** | 5GB ingestion | $2.50 |
| **Total** | | **~$10-15/month** |

> **Note**: Costs vary based on usage patterns. Training costs occur only when running training jobs.

### Cost Optimization Tips

✅ **Delete endpoints when not in use**:
```bash
aws sagemaker delete-endpoint --endpoint-name cbis-ddsm-serverless-endpoint
```

✅ **Use lifecycle policies for S3**:
```bash
# Archive old model artifacts to S3 Glacier after 90 days
```

✅ **Enable SageMaker automatic scaling**:
```python
# Serverless endpoints automatically scale to zero when idle
```

✅ **Clean up after experiments**:
```bash
# Execute cleanup cells in notebooks
# Run terraform destroy when done with dev environment
```

### Resource Cleanup

**Remove all infrastructure**:
```bash
cd infra
terraform destroy
# Type 'yes' to confirm
```

**Manual cleanup checklist**:
- [ ] Delete SageMaker endpoints
- [ ] Delete S3 bucket contents
- [ ] Terminate training jobs
- [ ] Delete CloudWatch log groups

---

##  Future Enhancements

### Planned Features

- [ ] **Model Drift Monitoring** ([05_monitoring_drift.ipynb](app/src/models/05_monitoring_drift.ipynb))
  - Automated data distribution tracking
  - Model performance degradation alerts
  - Scheduled retraining triggers

- [ ] **Complete MLOps Pipeline** ([06_mlops_pipeline.ipynb](app/src/models/06_mlops_pipeline.ipynb))
  - End-to-end automated workflow
  - CI/CD integration with GitHub Actions
  - Automated testing and validation

- [ ] **Multi-Class Classification**
  - Expand beyond binary classification
  - Detect specific lesion types
  - Severity scoring system

- [ ] **Real-Time Monitoring Dashboard**
  - Grafana/CloudWatch dashboard
  - Live inference metrics
  - Cost tracking and optimization alerts

- [ ] **Model Explainability**
  - Grad-CAM visualizations
  - Attention heatmaps
  - SHAP value analysis

- [ ] **API Gateway Integration**
  - RESTful API for inference
  - Authentication and rate limiting
  - Swagger documentation

### Contribution Ideas

We welcome contributions in these areas:
- Enhanced data augmentation techniques
- Alternative model architectures (EfficientNet, Vision Transformer)
- Integration with PACS systems
- Mobile application development
- Performance benchmarking

---

## Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**

   <a href="https://github.com/araujoeverton/breast-cancer-ml-classifier/fork">
       <img alt="Fork" title="Fork Repository" src="https://shields.io/badge/-FORK%20REPOSITORY-red.svg?&style=for-the-badge&logo=github&logoColor=white"/>
   </a>

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Commit your changes**
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Wait for review and feedback

### Code of Conduct

- Be respectful and inclusive
- Follow best practices for medical AI ethics
- Ensure HIPAA compliance in healthcare-related contributions
- Document all changes thoroughly

### Areas for Contribution

- Bug fixes and issue resolution
- Documentation improvements
- Test coverage expansion
- UI/UX enhancements
- Research and experimentation
- Internationalization

---

##  License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Usage Authorization

✅ **You are free to:**
- Use for educational and research purposes
- Fork and create derivative works

⚠️ **With the following conditions:**
- Provide attribution to the original author
- Include the MIT License in distributions
- Understand that this is provided "as-is" without warranty

> **Medical Disclaimer**: This project is for educational and research purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis without proper validation, regulatory approval, and oversight by qualified healthcare professionals.

---

## Contact & Support

### Author
**Everton Araujo**
- GitHub: [@araujoeverton](https://github.com/araujoeverton)
- Repository: [breast-cancer-ml-classifier](https://github.com/araujoeverton/breast-cancer-ml-classifier)

### Getting Help

- **Report Bugs**: [Open an issue](https://github.com/araujoeverton/breast-cancer-ml-classifier/issues/new)
- **Feature Requests**: [Start a discussion](https://github.com/araujoeverton/breast-cancer-ml-classifier/discussions)
- **Questions**: Use [GitHub Discussions](https://github.com/araujoeverton/breast-cancer-ml-classifier/discussions)

### Acknowledgments

- **CBIS-DDSM Dataset**: Curated Breast Imaging Subset of DDSM from Kaggle
- **ResNet Architecture**: He et al., "Deep Residual Learning for Image Recognition" (2015)
- **AWS SageMaker**: For providing robust MLOps infrastructure
- **Open Source Community**: For invaluable tools and libraries

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for advancing healthcare AI

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=araujoeverton.breast-cancer-ml-classifier)

</div>
