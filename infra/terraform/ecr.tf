locals {
  images = toset(["backend", "ml-service", "pipeline"])
}

resource "aws_ecr_repository" "app" {
  for_each             = local.images
  name                 = "${var.project}/${each.key}"
  image_tag_mutability = "IMMUTABLE" # a tag always means the same bits; no silent drift

  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  for_each   = aws_ecr_repository.app
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 20 images; storage is not free"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = { type = "expire" }
      }
    ]
  })
}
