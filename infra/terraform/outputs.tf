output "api_url" {
  description = "Public API entrypoint"
  value       = var.certificate_arn == "" ? "http://${aws_lb.main.dns_name}" : "https://${aws_lb.main.dns_name}"
}

output "dashboard_url" {
  value = "https://${aws_cloudfront_distribution.main.domain_name}"
}

output "frontend_bucket" {
  description = "aws s3 sync frontend/dist s3://<this>"
  value       = aws_s3_bucket.frontend.id
}

output "cloudfront_distribution_id" {
  description = "Needed for cache invalidation on deploy"
  value       = aws_cloudfront_distribution.main.id
}

output "ecr_repositories" {
  value = { for k, v in aws_ecr_repository.app : k => v.repository_url }
}

output "ecs_cluster" {
  value = aws_ecs_cluster.main.name
}

output "ecs_services" {
  value = {
    backend    = aws_ecs_service.backend.name
    ml_service = aws_ecs_service.ml.name
  }
}

output "db_endpoint" {
  value = aws_db_instance.main.address
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "db_password_secret_arn" {
  description = "Read with: aws secretsmanager get-secret-value --secret-id <this>"
  value       = aws_secretsmanager_secret.db.arn
}

output "model_artifact_filesystem" {
  value = aws_efs_file_system.models.id
}
