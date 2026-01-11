variable "aws_region" {
  description = "Region where the infra will be created"
  type        = string
}

variable "aws_account_id" {
  description = "AWS Account ID (used to ensure S3 bucket uniqueness)"
  type        = string
}

variable "project_name" {
  description = "Base project name"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev, prod, staging)"
  type        = string
}

variable "endpoint_name" {
  description = "Name of the SageMaker Endpoint"
  type        = string
}