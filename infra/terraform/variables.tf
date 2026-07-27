variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
  default = "prod"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "project" {
  type    = string
  default = "fantasyiq"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "availability_zone_count" {
  type    = number
  default = 2
  # Two AZs is the minimum for an ALB and for RDS multi-AZ failover.
}

# ---- Backend service sizing ----
variable "backend_cpu" {
  type    = number
  default = 512
}

variable "backend_memory" {
  type    = number
  default = 1024
}

variable "backend_desired_count" {
  type    = number
  default = 2
}

variable "backend_max_count" {
  type    = number
  default = 8
}

# ---- ML service sizing ----
variable "ml_cpu" {
  type    = number
  default = 1024 # inference is CPU-bound; this is the knob that moves p95
}

variable "ml_memory" {
  type    = number
  default = 2048
}

variable "ml_desired_count" {
  type    = number
  default = 2
}

# ---- Data stores ----
variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_allocated_storage" {
  type    = number
  default = 20
}

variable "db_multi_az" {
  type    = bool
  default = false # true in prod; doubles the cost, removes the single point of failure
}

variable "redis_node_type" {
  type    = string
  default = "cache.t4g.micro"
}

variable "active_model_version" {
  type    = string
  default = "xgboost_v1"
  # Changing this and applying is the model rollback path: no rebuild, no retrain.
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "alarm_email" {
  type        = string
  default     = ""
  description = "If set, SNS sends CloudWatch alarms here."
}
