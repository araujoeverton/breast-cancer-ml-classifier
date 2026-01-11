# Rule: Detect upload in 'entrada/' folder of the bucket
resource "aws_cloudwatch_event_rule" "s3_upload" {
  name        = "capture-s3-upload"
  description = "Trigger Lambda on S3 upload"

  event_pattern = jsonencode({
    source      = ["aws.s3"],
    detail-type = ["Object Created"],
    detail = {
      bucket = { name = [var.bucket_name] },
      object = { key = [{ prefix = "entrada/" }] }
    }
  })
}

# Target: The Lambda
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.s3_upload.name
  target_id = "SendToLambda"
  arn       = var.lambda_arn
}

# Permission: Allow EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = var.lambda_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.s3_upload.arn
}