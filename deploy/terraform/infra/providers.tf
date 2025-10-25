provider "google" {
  project = var.gcp_project
  region  = var.region
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}
