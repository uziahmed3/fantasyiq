# The weekly ETL runs as an EventBridge-scheduled Fargate task, not a long-lived
# container. It runs for a few minutes once a week; paying for an idle service the other
# 167 hours would be pure waste, and a scheduled task gets retries and CloudWatch
# failure metrics for free.

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${local.name}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name      = "pipeline"
    image     = "${local.registry}/${var.project}/pipeline:latest"
    essential = true
    command   = ["python", "-m", "run_weekly"]
    environment = concat(local.common_env, [
      { name = "ML_SERVICE_URL", value = "http://ml.${var.project}.internal:9000" },
      { name = "INGEST_SEASONS", value = "2023" },
      { name = "INGEST_POSITIONS", value = "WR,RB,TE,QB" },
    ])
    secrets = local.common_secrets
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.pipeline.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "etl"
      }
    }
  }])
}

resource "aws_cloudwatch_event_rule" "weekly_pipeline" {
  name                = "${local.name}-weekly-etl"
  description         = "Tuesday 09:00 UTC - after Monday Night Football"
  schedule_expression = "cron(0 9 ? * TUE *)"
}

resource "aws_cloudwatch_event_target" "weekly_pipeline" {
  rule     = aws_cloudwatch_event_rule.weekly_pipeline.name
  arn      = aws_ecs_cluster.main.arn
  role_arn = aws_iam_role.events.arn

  ecs_target {
    task_definition_arn = aws_ecs_task_definition.pipeline.arn
    launch_type         = "FARGATE"
    task_count          = 1
    platform_version    = "LATEST"

    network_configuration {
      subnets          = aws_subnet.private[*].id
      security_groups  = [aws_security_group.pipeline.id]
      assign_public_ip = false
    }
  }

  retry_policy {
    maximum_retry_attempts       = 2
    maximum_event_age_in_seconds = 3600
  }
}
