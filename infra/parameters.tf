# Save the bucket name to SSM
resource "aws_ssm_parameter" "bucket_name" {
  name        = "/${var.project_name}/${var.environment}/s3_bucket_name"
  description = "S3 Data Bucket Name"
  type        = "String"
  value       = module.s3.bucket_name
}

# Save the Role ARN to SSM
resource "aws_ssm_parameter" "role_arn" {
  name        = "/${var.project_name}/${var.environment}/sagemaker_role_arn"
  description = "SageMaker Execution Role ARN"
  type        = "String"
  value       = module.iam.sagemaker_role_arn
}

# Save the Endpoint Name to SSM
resource "aws_ssm_parameter" "endpoint_name" {
  name        = "/${var.project_name}/${var.environment}/endpoint_name"
  description = "Inference Endpoint Name"
  type        = "String"
  value       = "${var.endpoint_name}-${var.environment}"
}