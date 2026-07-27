locals {
  account_id = data.aws_caller_identity.current.account_id
  registry   = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"

  # Non-secret configuration shared by every service.
  common_env = [
    { name = "ENVIRONMENT", value = var.environment },
    { name = "POSTGRES_HOST", value = aws_db_instance.main.address },
    { name = "POSTGRES_PORT", value = tostring(aws_db_instance.main.port) },
    { name = "POSTGRES_DB", value = aws_db_instance.main.db_name },
    { name = "POSTGRES_USER", value = aws_db_instance.main.username },
    { name = "REDIS_URL", value = "redis://${aws_elasticache_replication_group.main.primary_endpoint_address}:6379/0" },
    { name = "ACTIVE_MODEL_VERSION", value = var.active_model_version },
  ]

  # Injected from Secrets Manager at task start - never baked into the image or the
  # task definition JSON.
  common_secrets = [
    { name = "POSTGRES_PASSWORD", valueFrom = aws_secretsmanager_secret.db.arn },
    { name = "JWT_SECRET_KEY", valueFrom = aws_secretsmanager_secret.jwt.arn },
  ]
}

resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${local.name}/backend"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "ml" {
  name              = "/ecs/${local.name}/ml-service"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/${local.name}/pipeline"
  retention_in_days = var.log_retention_days
}

# ------------------------------ Service discovery ------------------------------
# The backend needs a stable name for the ML service. Fargate task IPs change on every
# deploy, so a private DNS namespace is the alternative to a second load balancer.

resource "aws_service_discovery_private_dns_namespace" "main" {
  name = "${var.project}.internal"
  vpc  = aws_vpc.main.id
}

resource "aws_service_discovery_service" "ml" {
  name = "ml"

  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.main.id
    dns_records {
      type = "A"
      ttl  = 10
    }
    routing_policy = "MULTIVALUE"
  }

  health_check_custom_config {
    failure_threshold = 1
  }
}

# ------------------------------ ML service ------------------------------

resource "aws_ecs_task_definition" "ml" {
  family                   = "${local.name}-ml"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ml_cpu
  memory                   = var.ml_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name = "models"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.models.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.models.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([{
    name      = "ml-service"
    image     = "${local.registry}/${var.project}/ml-service:latest"
    essential = true
    portMappings = [{ containerPort = 9000, protocol = "tcp" }]
    environment = concat(local.common_env, [{ name = "MODEL_DIR", value = "/models" }])
    secrets     = local.common_secrets
    mountPoints = [{ sourceVolume = "models", containerPath = "/models", readOnly = false }]
    healthCheck = {
      command     = ["CMD-SHELL", "curl -fsS http://localhost:9000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 30
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ml.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ml"
      }
    }
  }])
}

resource "aws_ecs_service" "ml" {
  name            = "${local.name}-ml"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ml.arn
  desired_count   = var.ml_desired_count
  launch_type     = "FARGATE"

  enable_execute_command = true

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ml.id]
    assign_public_ip = false
  }

  service_registries {
    registry_arn = aws_service_discovery_service.ml.arn
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # a task that never becomes healthy rolls itself back
  }

  lifecycle {
    # CI updates the image tag; Terraform should not fight it back to :latest.
    ignore_changes = [task_definition]
  }
}

# ------------------------------ Backend API ------------------------------

resource "aws_ecs_task_definition" "backend" {
  family                   = "${local.name}-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.backend_cpu
  memory                   = var.backend_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "backend"
    image     = "${local.registry}/${var.project}/backend:latest"
    essential = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = concat(local.common_env, [
      { name = "ML_SERVICE_URL", value = "http://ml.${var.project}.internal:9000" },
      { name = "LOG_LEVEL", value = "INFO" },
    ])
    secrets = local.common_secrets
    healthCheck = {
      command     = ["CMD-SHELL", "curl -fsS http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 20
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.backend.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "backend" {
  name            = "${local.name}-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.backend_desired_count
  launch_type     = "FARGATE"

  enable_execute_command             = true
  health_check_grace_period_seconds  = 45

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.backend.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.backend.arn
    container_name   = "backend"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # Rolling deploy with no capacity dip: bring new tasks up before old ones go down.
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]

  lifecycle {
    ignore_changes = [task_definition]
  }
}

# ------------------------------ Autoscaling ------------------------------

resource "aws_appautoscaling_target" "backend" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.backend.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.backend_desired_count
  max_capacity       = var.backend_max_count
}

# Request-count scaling, not CPU: the API is IO-bound waiting on Postgres and the ML
# service, so CPU stays flat while latency climbs. Requests-per-target is the signal
# that actually correlates with user-visible slowness.
resource "aws_appautoscaling_policy" "backend_requests" {
  name               = "${local.name}-backend-rps"
  service_namespace  = aws_appautoscaling_target.backend.service_namespace
  resource_id        = aws_appautoscaling_target.backend.resource_id
  scalable_dimension = aws_appautoscaling_target.backend.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    target_value       = 600 # requests per task per minute
    scale_in_cooldown  = 300 # slow to shrink - avoids thrashing
    scale_out_cooldown = 60  # fast to grow

    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label = join("/", [
        aws_lb.main.arn_suffix,
        aws_lb_target_group.backend.arn_suffix,
      ])
    }
  }
}

resource "aws_appautoscaling_target" "ml" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ml.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.ml_desired_count
  max_capacity       = var.ml_desired_count * 4
}

# Inference genuinely is CPU-bound, so here CPU is the right signal.
resource "aws_appautoscaling_policy" "ml_cpu" {
  name               = "${local.name}-ml-cpu"
  service_namespace  = aws_appautoscaling_target.ml.service_namespace
  resource_id        = aws_appautoscaling_target.ml.resource_id
  scalable_dimension = aws_appautoscaling_target.ml.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    target_value       = 65
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
