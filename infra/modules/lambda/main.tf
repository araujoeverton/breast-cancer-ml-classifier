# Package the Python code
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = var.source_file_path
  output_path = "${path.module}/lambda_function.zip"
}

resource "aws_lambda_function" "inference" {
  filename      = data.archive_file.lambda_zip.output_path
  function_name = "${var.project_name}-processor"
  role          = var.iam_role_arn
  handler       = "inference_handler.lambda_handler"
  runtime       = "python3.9"
  timeout       = 30

  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      ENDPOINT_NAME = var.endpoint_name
    }
  }
}