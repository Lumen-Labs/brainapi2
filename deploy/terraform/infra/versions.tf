terraform {
  required_version = ">= 1.8.0"
  required_providers {
    google     = { source = "hashicorp/google", version = "~> 5.40" }
    cloudflare = { source = "cloudflare/cloudflare", version = "~> 4.40" }
    random     = { source = "hashicorp/random", version = "~> 3.6" }
  }
}
