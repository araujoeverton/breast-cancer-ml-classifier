resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name
  force_destroy = true # Be careful in prod! Allows deleting non-empty buckets.
}

# Enable EventBridge notifications
resource "aws_s3_bucket_notification" "main" {
  bucket      = aws_s3_bucket.main.id
  eventbridge = true
}