locals {
  workspace  = terraform.workspace   # if you choose one-workspace-per-user
  app_name   = "brainapi"
  machine    = var.plan == "pro" ? var.machine_type_pro : var.machine_type_small
  fqdn       = "${var.subdomain_prefix}-${var.user_slug}.${var.domain}"
  name_prefix= "${local.app_name}-${var.user_slug}"
  image_ref  = "${var.base_image}:${var.version}"
}

module "network" {
  source      = "./modules/network"
  name_prefix = local.name_prefix
  region      = var.region
  labels      = var.labels
}

module "secrets" {
  source      = "./modules/secrets"
  name_prefix = local.name_prefix
  ghcr_user   = var.ghcr_username
  ghcr_token  = var.ghcr_token
}

module "vm" {
  source          = "./modules/vm_docker"
  name_prefix     = local.name_prefix
  region          = var.region
  subnet_selflink = module.network.subnet_selflink
  machine_type    = local.machine
  fqdn            = local.fqdn
  image_ref       = local.image_ref
  ghcr_user       = var.ghcr_username
  ghcr_token      = var.ghcr_token
  labels          = var.labels
  enable_sql      = var.enable_sql
  # If SQL enabled, provide connection info via metadata/env
  db_conn_str     = var.enable_sql ? module.sql.conn_str : null
}

module "dns" {
  source   = "./modules/dns_cloudflare"
  zone_id  = var.zone_id
  name     = "${var.subdomain_prefix}-${var.user_slug}"
  domain   = var.domain
  target_ip= module.vm.public_ip
  ttl      = 120
}

module "sql" {
  count       = var.enable_sql ? 1 : 0
  source      = "./modules/sql"
  name_prefix = local.name_prefix
  region      = var.region
  labels      = var.labels
}
