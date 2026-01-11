# Locals to standardize names
locals {
  prefix = "${var.project_name}-${var.environment}"

  # Common tags to apply to all resources
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# 1. Create S3 Bucket
module "s3" {
  source       = "./modules/s3"
  bucket_name  = "${local.prefix}-data-${var.aws_account_id}"
}

# 2. Create IAM Roles
module "iam" {
  source        = "./modules/iam"
  project_name  = local.prefix
  s3_bucket_arn = module.s3.bucket_arn
}

# 3. Create Lambda Function
module "lambda" {
  source           = "./modules/lambda"
  project_name     = local.prefix
  iam_role_arn     = module.iam.lambda_role_arn
  endpoint_name    = "${var.endpoint_name}-${var.environment}" # Endpoint also gets a suffix
  source_file_path = "${path.module}/src/inference_handler.py"
}

# 4. Configure EventBridge
module "eventbridge" {
  source        = "./modules/eventbridge"
  bucket_name   = module.s3.bucket_name
  lambda_arn    = module.lambda.function_arn
  lambda_name   = module.lambda.function_name
}