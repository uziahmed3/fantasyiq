terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }

  # Remote state with locking. Create the bucket + table once, out of band, then
  # `terraform init -backend-config=backend.hcl`.
  backend "s3" {
    key          = "fantasyiq/terraform.tfstate"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "fantasyiq"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
