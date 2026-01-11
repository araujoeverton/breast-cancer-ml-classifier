# 1. SAGEMAKER ROLE (Used by Notebooks & Training Jobs)
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

# Standard SageMaker Permissions
resource "aws_iam_role_policy_attachment" "sm_full" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# S3 Access (Required for Training Data)
resource "aws_iam_role_policy_attachment" "sm_s3" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess" # Restrict this in production!
}

# --- SSM Parameter Store Access ---
resource "aws_iam_policy" "ssm_read_policy" {
  name = "${var.project_name}-ssm-read"
  description = "Allows SageMaker to read configuration parameters from SSM"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "ssm:GetParameter"
      # Allows reading any parameter that starts with the project name
      Resource = "arn:aws:ssm:*:*:parameter/${var.project_name}/*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_attach" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.ssm_read_policy.arn
}

# 2. LAMBDA ROLE (Used by Inference/EventBridge)
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Action = "sts:AssumeRole",
      Effect = "Allow",
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

# Custom Policy: Read S3 + Call SageMaker + Logs
resource "aws_iam_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = ["s3:GetObject", "s3:ListBucket"],
        Resource = [var.s3_bucket_arn, "${var.s3_bucket_arn}/*"]
      },
      {
        Effect = "Allow",
        Action = ["sagemaker:InvokeEndpoint"],
        Resource = "*"
      },
      {
        Effect = "Allow",
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_attach" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_policy.arn
}