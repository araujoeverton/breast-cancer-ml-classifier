# --- 1. S3 Outputs (Critical for Notebooks 01, 02, 03) ---
output "s3_bucket_name" {
  description = "The exact name of the created S3 bucket. Copy this to your Python notebooks."
  value       = module.s3.bucket_name
}

output "s3_bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = module.s3.bucket_arn
}

# --- 2. IAM Outputs (Critical for SageMaker Training) ---
output "sagemaker_role_arn" {
  description = "The IAM Role ARN that SageMaker will use. Replace 'get_execution_role()' with this in production."
  value       = module.iam.sagemaker_role_arn
}

# --- 3. Lambda Outputs (For Debugging) ---
output "lambda_function_name" {
  description = "The name of the inference Lambda function"
  value       = module.lambda.function_name
}

output "endpoint_target" {
  description = "The SageMaker Endpoint name configured in the Lambda environment"
  value       = var.endpoint_name
}